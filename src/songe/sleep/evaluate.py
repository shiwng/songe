import json
import random
import re
import shutil
from pathlib import Path
from statistics import mean

from ..steps import digest
from .prompts import paired_prompts

XML_FUNCTION = re.compile(r"^\s*<function=([^>\s]+)>(.*)</function>\s*$", re.S)
XML_PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def xml_call(segment, tools):
    """A tool call in the XML form some models write inside <tool_call>
    (LFM3, Qwen3-Coder): <function=name><parameter=key>value</parameter>...
    Values are text; a parameter the schema types as non-string is read as
    JSON. None when the segment is not in this form."""
    found = XML_FUNCTION.match(segment)
    if not found:
        return None
    name, body = found.groups()
    schema = next(
        (
            (t.get("function", {}).get("parameters") or {}).get("properties") or {}
            for t in tools or []
            if t.get("function", {}).get("name") == name
        ),
        {},
    )
    arguments = {}
    for key, value in XML_PARAMETER.findall(body):
        kind = (schema.get(key) or {}).get("type")
        if kind not in (None, "string"):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        arguments[key] = value
    leftover = XML_PARAMETER.sub("", body).strip()
    if leftover:
        return None  # text outside the parameters: not a clean call
    return {"name": name, "arguments": arguments}


def health(text, tokenizer, tools, tool_step, repetition_ngram=4):
    calls = []
    valid = not tool_step
    supported, malformed = True, False
    if tool_step:
        parser = getattr(tokenizer, "tool_parser", None)
        if parser:
            # Use the installed parser and its own family-specific delimiters.
            import importlib

            module = importlib.import_module(parser.__module__)
            start = getattr(module, "tool_call_start", "<tool_call>")
            end = getattr(module, "tool_call_end", "</tool_call>")
            segments = re.findall(
                re.escape(start) + r"(.*?)" + re.escape(end), text, re.S
            )
            for segment in segments:
                try:
                    result = parser(segment, tools)
                    calls.extend(result if isinstance(result, list) else [result])
                except (ValueError, TypeError, KeyError):
                    malformed = True
            malformed |= text.count(start) != len(segments) or text.count(end) != len(
                segments
            )
        if not parser:
            # Recognize the complete protocol, never JSON embedded in prose.
            # Unknown model-specific framing cannot certify tool health.
            start, end = "<tool_call>", "</tool_call>"
            segments = re.findall(start + r"(.*?)" + end, text, re.S)
            template = getattr(tokenizer, "chat_template", "") or ""
            if isinstance(template, dict):
                template = "\n".join(template.values())
            # Support belongs to the tokenizer protocol, not to whether this
            # sample complied. A missing/truncated call in a known protocol is
            # invalid and still participates in the before/after validity gate.
            supported = (
                (start in template and end in template)
                or bool(segments)
                or start in text
                or end in text
            )
            malformed = text.count(start) != len(segments) or text.count(end) != len(
                segments
            )
            if not supported:
                try:
                    value = json.loads(text)
                    segments = [text] if isinstance(value, (dict, list)) else []
                    supported = bool(segments)
                except ValueError:
                    pass
            for segment in segments:
                try:
                    value = json.loads(segment)
                    for call in value if isinstance(value, list) else [value]:
                        fn = call.get("function", call)
                        if "name" in fn and "arguments" in fn:
                            calls.append(fn)
                        else:
                            malformed = True
                except (ValueError, TypeError, AttributeError):
                    xml = xml_call(segment, tools)
                    if xml is None:
                        malformed = True
                    else:
                        calls.append(xml)
        from ..steps import valid_action

        valid = (
            bool(calls)
            and not malformed
            and valid_action(
                {
                    "tool_calls": [
                        c if "function" in c else {"function": c} for c in calls
                    ]
                },
                tools,
            )
        )
    words = text.split()
    phrases = [
        tuple(words[i : i + repetition_ngram])
        for i in range(len(words) - repetition_ngram + 1)
    ]
    repeated = 1 - len(set(phrases)) / len(phrases) if phrases else 0.0
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return {
        "calls": calls,  # parsed tool calls, for before/after examples
        "tool_valid": int(valid),
        "tool_supported": int(supported),
        "empty": int(not visible),
        "repetition": repeated,
    }


def references(base, steps, *, cache=None):
    """The original model's sampled answer for each gate step.

    cache: a ReferenceCache. The answer depends only on the original weights,
    the prompt and the answer length, so later Dreams reuse it instead of
    sampling again (sampling is most of a reference's cost)."""
    if any(s.get("split") not in {"heldout", "anchor", "drift"} for s in steps):
        raise ValueError(
            "Gate references require held-out real data, fixed anchors or the drift set"
        )
    refs = []
    for step in steps:
        getattr(base, "boundary", base.check)()
        prompt, teacher = paired_prompts(
            base.tokenizer, step, base.config, check_length=False
        )
        y = cache.get(prompt, base.config.response_tokens) if cache else None
        if y is None:
            with base.base():
                y, _, _ = base.sample(prompt, seed=7613)
            if cache:
                cache.put(prompt, base.config.response_tokens, y)
        refs.append(
            {
                "id": step["id"],
                "conversation": step["conversation"],
                "prompt": prompt,
                "teacher": teacher,
                "y": y,
                "tools": step.get("tools", []),
                "tool_step": bool(step.get("action", {}).get("tool_calls")),
            }
        )
    return refs


class ReferenceCache:
    """Reference answers kept across Dreams for one original model.

    One folder per base fingerprint; folders for other fingerprints (an
    older base) are removed when this one is first written, so the cache
    never outlives the weights it describes."""

    def __init__(self, root, fingerprint):
        self.root, self.fingerprint = Path(root), fingerprint
        self.folder = self.root / fingerprint[:24]
        self.pruned = False

    def path(self, prompt, length):
        return self.folder / (digest([self.fingerprint, length, list(prompt)]) + ".json")

    def get(self, prompt, length):
        try:
            return json.loads(self.path(prompt, length).read_text(encoding="utf-8"))["y"]
        except (OSError, ValueError, KeyError):
            return None

    def put(self, prompt, length, y):
        if not self.pruned:
            self.pruned = True
            if self.root.is_dir():
                for other in self.root.iterdir():
                    if other.is_dir() and other != self.folder:
                        shutil.rmtree(other, ignore_errors=True)
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.path(prompt, length)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"y": [int(t) for t in y]}), encoding="utf-8")
        temporary.replace(path)


def kl_between(base, candidate, prompt, teacher_prompt, y, *, hindsight=False, views=None):
    # Held-out hindsight KL uses the same teacher as training (the version
    # the cycle started from); retention drift stays against the original base.
    # views: the reference's reads kept across versions (see kl_from).
    extra = {} if views is None else {"views": views}
    return candidate.kl_from(base, prompt, teacher_prompt, y, hindsight=hindsight, **extra)


def interval(values_by_cluster, confidence, bootstraps=2000):
    """Two-sided percentile bootstrap interval of the mean, resampling
    clusters (conversations), as the gate's noise margin does."""
    means = [mean(v) for v in values_by_cluster.values()]
    rng = random.Random(7613)
    draws = sorted(mean(rng.choices(means, k=len(means))) for _ in range(bootstraps))
    tail = int((1 - confidence) / 2 * bootstraps)
    return draws[tail], draws[bootstraps - 1 - tail]


def drift_choice(general, drift_values, drift_refs, config):
    """Which "not broken" measure gates (docs/CONTRACTS.md, Integration
    decisions): the drift set once it has min_drift conversations and its
    interval is narrower than the limit; the fixed anchors until then."""
    info = {"general": general, "source": "general", "drift_set": None}
    if not drift_refs:
        info["reason"] = "no confirmed successes yet"
        return general, info
    clusters = {}
    for r, value in zip(drift_refs, drift_values, strict=True):
        clusters.setdefault(r["conversation"], []).append(value)
    info["drift_set"] = mean(drift_values)
    info["conversations"] = len(clusters)
    if len(clusters) < config.min_drift:
        info["reason"] = f"{len(clusters)} of {config.min_drift} conversations"
        return general, info
    low, high = interval(clusters, config.gate_confidence)
    info["interval"] = [low, high]
    if high - low >= config.general_drift_limit:
        info["reason"] = "interval wider than the drift limit"
        return general, info
    info["source"], info["reason"] = "drift", "enough confirmed successes"
    return info["drift_set"], info


def evaluate(
    base,
    candidate,
    refs,
    general_refs,
    *,
    drift_refs=(),
    checkpoint=None,
    commit=lambda: None,
    identical=False,
    views=None,
):
    """Gate rows for one version against the frozen base.

    identical: this version is the base itself (the first Dream's
    incumbent), so its drift from the base is zero by definition and is not
    measured; held-out rows still are.

    views: a dict shared by the versions of one gate, so the reference's
    read of each moment (the same for every version) is done once."""
    checkpoint = {} if checkpoint is None else checkpoint
    # The drift set first: the same measure on real contexts where the base
    # was right (confirmed successes). Never trained on. When it can decide
    # (drift_choice), the bundled prompts are not needed and not measured.
    drift_set = checkpoint.setdefault("drift_set", [])
    for r in list(drift_refs)[len(drift_set) :]:
        if identical:
            drift_set.append(0.0)
            continue
        getattr(candidate, "boundary", candidate.check)()
        drift_set.append(kl_between(base, candidate, r["prompt"], r["prompt"], r["y"], views=views))
        commit()
    drifts = checkpoint.setdefault("drifts", [])
    _, decided = drift_choice(None, drift_set, list(drift_refs), candidate.config)
    if decided["source"] != "drift":
        # The bundled prompts, with the same samples for every version: a
        # candidate cannot pass by choosing easier generated sequences.
        if callable(general_refs):
            general_refs = general_refs()  # computed only when needed
        for r in general_refs[len(drifts) :]:
            if identical:
                drifts.append(0.0)
                continue
            getattr(candidate, "boundary", candidate.check)()
            drifts.append(kl_between(base, candidate, r["prompt"], r["prompt"], r["y"], views=views))
            commit()
    commit()
    drift, checkpoint["drift_check"] = drift_choice(
        mean(drifts) if drifts else None, drift_set, list(drift_refs), candidate.config
    )
    rows = checkpoint.setdefault("rows", [])
    for r in refs[len(rows) :]:
        getattr(candidate, "boundary", candidate.check)()
        kl = kl_between(
            base, candidate, r["prompt"], r["teacher"], r["y"], hindsight=True, views=views
        )
        _, text, _ = candidate.sample(r["prompt"], seed=7613)
        rows.append(
            {
                "id": r["id"],
                "conversation": r["conversation"],
                "kl": kl,
                "drift": drift,
                "tool_step": r["tool_step"],
                **{
                    k: v
                    for k, v in health(
                        text,
                        candidate.tokenizer,
                        r["tools"],
                        r["tool_step"],
                        candidate.config.repetition_ngram,
                    ).items()
                    if k != "calls"  # gate rows keep measurements only
                },
            }
        )
        commit()
    return rows
