from __future__ import annotations

import torch

from rag.generation.generator import (
    GenerationBackend,
    Generator,
)


def create_generator(
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
) -> Generator:
    return Generator(
        model_id=model_id,
        backend=backend,
        device_map=device_map,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
        trust_remote_code=trust_remote_code,
    )