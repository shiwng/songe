"""Array-API divergence reference shared by backends and CPU-only tests."""

import math


def divergence(
    mx,
    student,
    teacher,
    kind="forward_kl",
    beta=0.5,
    top_k=0,
    reduction="mean",
    clip=0.0,
):
    """KL(teacher || student) (OPSD/SDFT default), KL(student || teacher), or
    beta*KL(teacher||mix)+(1-beta)*KL(student||mix).

    clip > 0 bounds each vocabulary entry's forward-KL contribution,
    sum_v min(p_t(v) * (log p_t(v) - log p_s(v)), clip), as OPSD does.

    Optional top-k retains the teacher's top tokens AND one exact aggregate tail
    bucket; omitting the tail would silently change the objective.
    """
    s = student.astype(mx.float32)
    t = mx.stop_gradient(teacher.astype(mx.float32))
    s = s - mx.logsumexp(s, axis=-1, keepdims=True)
    t = t - mx.logsumexp(t, axis=-1, keepdims=True)
    if top_k and top_k < t.shape[-1]:
        idx = mx.argpartition(t, kth=t.shape[-1] - top_k, axis=-1)[..., -top_k:]
        s_top, t_top = (
            mx.take_along_axis(s, idx, axis=-1),
            mx.take_along_axis(t, idx, axis=-1),
        )
        s_tail = mx.log(mx.maximum(1 - mx.exp(s_top).sum(-1, keepdims=True), 1e-12))
        t_tail = mx.log(mx.maximum(1 - mx.exp(t_top).sum(-1, keepdims=True), 1e-12))
        s, t = mx.concatenate([s_top, s_tail], -1), mx.concatenate([t_top, t_tail], -1)
    if kind == "forward_kl":
        terms = mx.exp(t) * (t - s)
        if clip > 0:
            terms = mx.minimum(terms, clip)
        value = terms.sum(-1)
        return value if reduction == "none" else value.mean()
    if kind == "reverse_kl":
        value = (mx.exp(s) * (s - t)).sum(-1)
        return value if reduction == "none" else value.mean()
    mix = mx.logaddexp(math.log(beta) + t, math.log(1 - beta) + s)
    value = (beta * mx.exp(t) * (t - mix) + (1 - beta) * mx.exp(s) * (s - mix)).sum(-1)
    return value if reduction == "none" else value.mean()


def view_key(hindsight, teacher_prompt, y):
    """Which reference view of a moment a kept one is (kl_from's views):
    whose weights read it (the teacher, or the frozen base), the context and
    the answer."""
    return (bool(hindsight), len(teacher_prompt), hash(tuple(teacher_prompt)), tuple(y))
