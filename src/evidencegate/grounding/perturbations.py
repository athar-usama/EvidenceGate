"""Stochastic perturbation sampler that turns one candidate into K independent decodes.

This is the input-side half of Consensus-Gated Grounding: instead of asking the
grounding model once, we ask it K times under crop jitter, scale jitter, and
prompt paraphrase, each with its own sampling temperature. A claim only earns
trust if these independent asks agree.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image

from evidencegate.geometry import Box
from evidencegate.grounding.prompts import prompts_for


@dataclass(frozen=True)
class Perturbation:
    crop_image: Image.Image
    origin: tuple[float, float]  # (x_offset, y_offset) of the crop in original-image pixel coords
    prompt: str
    temperature: float


DEFAULT_CONTEXT_LADDER = (48, 80, 120, 160, 210, 260)


def sample_perturbations(
    image: Image.Image,
    box: Box,
    family: str,
    k: int,
    rng: random.Random,
    base_margin: float = 0.6,
    translate_jitter: float = 0.15,
    min_temperature: float = 0.6,
    max_temperature: float = 1.2,
    context_ladder: tuple[int, ...] = DEFAULT_CONTEXT_LADDER,
) -> list[Perturbation]:
    """Each of the K runs gets a genuinely different amount of surrounding context, stepping
    through `context_ladder` (in pixels) rather than jittering around one fixed crop size.

    This matters: most candidates here are only a few pixels wide (microaneurysm scale), so a
    single fixed crop floor collapses every run into nearly the same image; consensus across
    K near-identical inputs is not a real independence check, it just measures whether the model
    is deterministic. Stepping through genuinely different fields of view is what lets a real,
    compact lesion stay consistently localized while a spurious classical-CV trigger (a vessel
    edge, an illumination artifact) is freer to drift to a different salient feature as more of
    the surrounding retina comes into view.
    """
    prompts = prompts_for(family)
    width, height = image.size
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    own_padded_size = max(bw, bh) * (1 + 2 * base_margin)

    perturbations: list[Perturbation] = []
    for i in range(k):
        crop_size = max(context_ladder[i % len(context_ladder)], own_padded_size)
        cx, cy = box.center()
        dx = rng.uniform(-translate_jitter, translate_jitter) * crop_size
        dy = rng.uniform(-translate_jitter, translate_jitter) * crop_size
        half = crop_size / 2
        shifted = Box(cx + dx - half, cy + dy - half, cx + dx + half, cy + dy + half).clip(width, height)

        x1, y1, x2, y2 = shifted.as_int_tuple()
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2))

        prompt = prompts[i % len(prompts)]
        temperature = rng.uniform(min_temperature, max_temperature)
        perturbations.append(
            Perturbation(crop_image=crop, origin=(float(x1), float(y1)), prompt=prompt, temperature=temperature)
        )
    return perturbations
