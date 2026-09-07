"""Thin wrapper around Florence-2 for phrase grounding, used as the sole VLM in the pipeline.

Deliberately zero-shot: no medical fine-tuning. The point of the project is the
trust layer around a general-purpose grounding model, not a medical detector,
so a stronger domain model would undersell the contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from evidencegate.geometry import Box

TASK_TOKEN = "<CAPTION_TO_PHRASE_GROUNDING>"


@dataclass(frozen=True)
class GroundingHit:
    box: Box
    label: str


class Grounder(Protocol):
    def ground(self, image: Image.Image, phrase: str, temperature: float) -> list[GroundingHit]: ...


class Florence2Grounder:
    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large-ft",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=self.dtype, attn_implementation="sdpa"
        ).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    @torch.inference_mode()
    def ground(self, image: Image.Image, phrase: str, temperature: float) -> list[GroundingHit]:
        prompt = TASK_TOKEN + phrase
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.dtype)

        generate_kwargs = dict(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            early_stopping=False,
        )
        if temperature and temperature > 0.05:
            generate_kwargs.update(do_sample=True, temperature=float(temperature), num_beams=1, top_p=0.9)
        else:
            generate_kwargs.update(do_sample=False, num_beams=3)

        generated_ids = self.model.generate(**generate_kwargs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=TASK_TOKEN, image_size=(image.width, image.height)
        )
        result = parsed.get(TASK_TOKEN, {})
        bboxes = result.get("bboxes", [])
        labels = result.get("labels", [phrase] * len(bboxes))

        hits = []
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            hits.append(GroundingHit(box=Box(float(x1), float(y1), float(x2), float(y2)), label=str(label)))
        return hits
