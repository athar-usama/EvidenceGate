"""Visual-description prompt bank for the grounding stage.

Florence-2 is a general-purpose, zero-shot grounding model with no medical
fine-tuning — deliberately so (see docs/METHOD.md). It grounds on visual
appearance, not clinical vocabulary, so the prompt bank describes what a
lesion *looks like* rather than naming it clinically. Each family has several
paraphrases; sampling different paraphrases across the K consensus runs is
one of the two sources of decoding diversity (the other is crop/scale jitter).
"""

from __future__ import annotations

PROMPT_BANK: dict[str, list[str]] = {
    "red": [
        "a tiny dark red dot",
        "a small round reddish spot",
        "a pinpoint dark red mark",
        "an irregular dark red blotch",
        "a small reddish-brown lesion",
    ],
    "bright": [
        "a sharp-edged yellow-white patch",
        "a bright yellowish deposit",
        "a crisp pale yellow spot",
        "a fuzzy pale white patch",
        "a soft cloudy whitish spot",
    ],
}

TASK_TOKEN = "<CAPTION_TO_PHRASE_GROUNDING>"


def prompts_for(family: str) -> list[str]:
    return PROMPT_BANK[family]
