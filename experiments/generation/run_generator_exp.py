from __future__ import annotations

import gc
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import torch
import yaml

from experiments.generation.efficiency import (
    measure_generation,
)
from rag.evaluation.models import GoldenQuestion
from rag.generation.factory import create_generator
from rag.generation.generator import (
    GenerationBackend,
    Generator,
)
from rag.models import Chunk


def load_generation_config(
    path: str | Path,
) -> dict:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(config, dict):
        raise ValueError(
            "generation config must "
            "contain a mapping"
        )

    if "common" not in config:
        raise ValueError(
            "generation config must contain "
            "'common'"
        )

    if "models" not in config:
        raise ValueError(
            "generation config must contain "
            "'models'"
        )

    return config


def create_generator_for_configuration(
    *,
    configuration: str,
    config: dict,
    device_map: str = "auto",
    torch_dtype: torch.dtype = (
        torch.bfloat16
    ),
) -> Generator:
    common = config["common"]

    model_config = config[
        "models"
    ][configuration]

    return create_generator(
        model_id=str(
            model_config["model_id"]
        ),
        backend=cast(
            GenerationBackend,
            model_config["backend"],
        ),
        device_map=device_map,
        torch_dtype=torch_dtype,
        max_new_tokens=int(
            model_config.get(
                "max_new_tokens",
                common.get(
                    "max_new_tokens",
                    256,
                ),
            )
        ),
        do_sample=bool(
            model_config.get(
                "do_sample",
                common.get(
                    "do_sample",
                    False,
                ),
            )
        ),
        temperature=model_config.get(
            "temperature",
            common.get(
                "temperature"
            ),
        ),
        top_p=model_config.get(
            "top_p",
            common.get(
                "top_p"
            ),
        ),
        enable_thinking=model_config.get(
            "enable_thinking"
        ),
        trust_remote_code=bool(
            model_config.get(
                "trust_remote_code",
                False,
            )
        ),
    )


def run_generation_model(
    *,
    generator: Generator,
    questions: Sequence[
        GoldenQuestion
    ],
    contexts: Mapping[
        str,
        Sequence[Chunk],
    ],
    configuration: str,
    context_type: str,
    warmup_queries: int = 3,
) -> list[dict]:
    if context_type not in {
        "gold",
        "retrieved",
    }:
        raise ValueError(
            "context_type must be "
            "'gold' or 'retrieved'"
        )

    _run_warmup(
        generator=generator,
        questions=questions,
        contexts=contexts,
        n_queries=warmup_queries,
    )

    records: list[dict] = []

    for index, question in enumerate(
        questions,
        start=1,
    ):
        chunks = list(
            contexts.get(
                question.question_id,
                [],
            )
        )

        context_texts = [
            chunk.text
            for chunk in chunks
        ]

        raw_output, efficiency = (
            measure_generation(
                generator=generator,
                question=question.question,
                context=context_texts,
            )
        )

        parsed = parse_generation_output(
            raw_output,
            n_fragments=len(chunks),
        )

        used_chunk_ids = [
            chunks[
                fragment_index - 1
            ].chunk_id
            for fragment_index
            in parsed["used_fragment_indices"]
        ]

        record = {
            "question_id": (
                question.question_id
            ),
            "configuration": (
                configuration
            ),
            "model_id": (
                generator.model_id
            ),
            "context_type": (
                context_type
            ),
            "question": (
                question.question
            ),
            "answer": (
                parsed["answer"]
            ),
            "raw_output": (
                raw_output
            ),
            "output_format_valid": (
                parsed[
                    "output_format_valid"
                ]
            ),
            "parse_error": (
                parsed["parse_error"]
            ),
            "used_fragment_indices": (
                parsed[
                    "used_fragment_indices"
                ]
            ),
            "used_chunk_ids": (
                used_chunk_ids
            ),
            "context": [
                {
                    "position": position,
                    "chunk_id": (
                        chunk.chunk_id
                    ),
                    "document_id": (
                        chunk.document_id
                    ),
                    "text": chunk.text,
                    "start_offset": (
                        chunk.start_offset
                    ),
                    "end_offset": (
                        chunk.end_offset
                    ),
                }
                for position, chunk
                in enumerate(
                    chunks,
                    start=1,
                )
            ],
            **efficiency,
        }

        records.append(
            record
        )

        print(
            f"[{index}/{len(questions)}] "
            f"{question.question_id}"
        )

    return records


def run_generation_experiment(
    *,
    questions: Sequence[
        GoldenQuestion
    ],
    contexts: Mapping[
        str,
        Sequence[Chunk],
    ],
    context_type: str,
    config_path: str | Path = (
        "configs/generation.yaml"
    ),
    output_dir: str | Path = (
        "results/generation/raw"
    ),
    device_map: str = "auto",
    torch_dtype: torch.dtype = (
        torch.bfloat16
    ),
) -> list[dict]:
    config = load_generation_config(
        config_path
    )

    output_dir = Path(
        output_dir
    )

    all_records: list[dict] = []

    for configuration in config[
        "models"
    ]:
        print()
        print(
            "================================="
        )
        print(
            f"Model: {configuration}"
        )
        print(
            f"Context: {context_type}"
        )
        print(
            "================================="
        )

        generator = (
            create_generator_for_configuration(
                configuration=configuration,
                config=config,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
        )

        try:
            records = run_generation_model(
                generator=generator,
                questions=questions,
                contexts=contexts,
                configuration=(
                    configuration
                ),
                context_type=(
                    context_type
                ),
                warmup_queries=int(
                    config["common"].get(
                        "warmup_queries",
                        3,
                    )
                ),
            )

            save_raw_records(
                records=records,
                output_path=(
                    output_dir
                    / (
                        f"{configuration}_"
                        f"{context_type}.jsonl"
                    )
                ),
            )

            all_records.extend(
                records
            )

        finally:
            del generator

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_records


def parse_generation_output(
    raw_output: str,
    *,
    n_fragments: int,
) -> dict:
    cleaned = raw_output.strip()

    fenced_match = re.search(
        r"```(?:json)?\s*(.*?)```",
        cleaned,
        flags=re.DOTALL
        | re.IGNORECASE,
    )

    if fenced_match is not None:
        cleaned = (
            fenced_match
            .group(1)
            .strip()
        )

    first_brace = cleaned.find(
        "{"
    )
    last_brace = cleaned.rfind(
        "}"
    )

    if (
        first_brace != -1
        and last_brace != -1
        and last_brace > first_brace
    ):
        candidate = cleaned[
            first_brace:
            last_brace + 1
        ]
    else:
        candidate = cleaned

    try:
        data = json.loads(
            candidate
        )

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "Output JSON is not "
                "an object"
            )

        answer = data.get(
            "answer"
        )

        used_fragments = data.get(
            "used_fragments"
        )

        if not isinstance(
            answer,
            str,
        ):
            raise ValueError(
                "'answer' must be "
                "a string"
            )

        if not isinstance(
            used_fragments,
            list,
        ):
            raise ValueError(
                "'used_fragments' "
                "must be a list"
            )

        parsed_indices: list[int] = []
        seen: set[int] = set()

        for value in used_fragments:
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    int,
                )
            ):
                raise ValueError(
                    "Fragment indices must "
                    "be integers"
                )

            if not (
                1 <= value <= n_fragments
            ):
                raise ValueError(
                    "Fragment index out "
                    f"of range: {value}"
                )

            if value not in seen:
                seen.add(value)
                parsed_indices.append(
                    value
                )

        return {
            "answer": answer.strip(),
            "used_fragment_indices": (
                parsed_indices
            ),
            "output_format_valid": True,
            "parse_error": None,
        }

    except (
        json.JSONDecodeError,
        ValueError,
    ) as exc:

        return {
            "answer": raw_output.strip(),
            "used_fragment_indices": [],
            "output_format_valid": False,
            "parse_error": str(exc),
        }


def save_raw_records(
    *,
    records: Sequence[dict],
    output_path: str | Path,
) -> None:
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def _run_warmup(
    *,
    generator: Generator,
    questions: Sequence[
        GoldenQuestion
    ],
    contexts: Mapping[
        str,
        Sequence[Chunk],
    ],
    n_queries: int,
) -> None:
    if n_queries <= 0:
        return

    warmup_count = min(
        n_queries,
        len(questions),
    )

    print(
        f"Warm-up: {warmup_count} queries"
    )

    for question in questions[
        :warmup_count
    ]:
        chunks = contexts.get(
            question.question_id,
            [],
        )

        generator.generate(
            question=question.question,
            context=[
                chunk.text
                for chunk in chunks
            ],
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()