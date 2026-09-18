from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from rag.generation.prompts import (
    build_rag_prompt,
)


GenerationBackend = Literal[
    "causal_lm",
    "multimodal_lm",
]


class Generator:

    def __init__(
        self,
        *,
        model_id: str,
        backend: GenerationBackend,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        enable_thinking: bool | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        if not model_id.strip():
            raise ValueError(
                "model_id must not be empty"
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0"
            )

        if (
            temperature is not None
            and temperature <= 0
        ):
            raise ValueError(
                "temperature must be greater than 0"
            )

        if (
            top_p is not None
            and not 0.0 < top_p <= 1.0
        ):
            raise ValueError(
                "top_p must be in (0, 1]"
            )

        self._model_id = model_id
        self._backend = backend

        self._max_new_tokens = (
            max_new_tokens
        )
        self._do_sample = (
            do_sample
        )
        self._temperature = (
            temperature
        )
        self._top_p = (
            top_p
        )
        self._enable_thinking = (
            enable_thinking
        )

        if backend == "causal_lm":
            self._processor = (
                AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=(
                        trust_remote_code
                    ),
                )
            )

            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=(
                        trust_remote_code
                    ),
                )
            )

        elif backend == "multimodal_lm":
            from transformers import (
                AutoModelForImageTextToText,
                )

            self._processor = (
                AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote_code,
                )
            )

            self._model = (
                AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        )

        else:
            raise ValueError(
                "Unsupported generation backend: "
                f"{backend}"
            )

        self._model.eval()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend(self) -> GenerationBackend:
        return self._backend

    def count_tokens(
        self,
        text: str,
    ) -> int:
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        tokenizer = getattr(
            self._processor,
            "tokenizer",
            self._processor,
        )

        encoded = tokenizer(
            text,
            add_special_tokens=False,
        )

        input_ids = encoded[
            "input_ids"
        ]

        if (
            input_ids
            and isinstance(
                input_ids[0],
                list,
            )
        ):
            input_ids = input_ids[0]

        return len(input_ids)

    def generate(
        self,
        *,
        question: str,
        context: Sequence[str],
    ) -> str:
        prompt = build_rag_prompt(
            question=question,
            context=context,
        )

        messages = self._build_messages(
            prompt
        )

        inputs = self._prepare_inputs(
            messages
        )

        input_length = (
            inputs["input_ids"].shape[-1]
        )

        with torch.inference_mode():
            output_ids = (
                self._model.generate(
                    **inputs,
                    **self._generation_kwargs(),
                )
            )

        generated_ids = output_ids[
            0,
            input_length:,
        ]

        answer = self._processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return answer.strip()

    def _build_messages(
        self,
        prompt: str,
    ) -> list[dict]:
        if self._backend == "causal_lm":
            return [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

    def _prepare_inputs(
        self,
        messages: list[dict],
    ):
        template_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }

        if self._enable_thinking is not None:
            template_kwargs[
                "enable_thinking"
            ] = self._enable_thinking

        inputs = (
            self._processor.apply_chat_template(
                messages,
                **template_kwargs,
            )
        )

        input_device = next(
            self._model.parameters()
        ).device

        return inputs.to(
            input_device
        )

    def _generation_kwargs(
        self,
    ) -> dict:
        kwargs = {
            "max_new_tokens": (
                self._max_new_tokens
            ),
            "do_sample": (
                self._do_sample
            ),
        }

        if self._do_sample:
            if self._temperature is not None:
                kwargs[
                    "temperature"
                ] = self._temperature

            if self._top_p is not None:
                kwargs[
                    "top_p"
                ] = self._top_p

        return kwargs