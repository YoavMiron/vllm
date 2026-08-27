# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
import numpy as np


class MoeActivationTraceWindow(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):  # type: ignore[call-arg]
    """Routed-expert decisions for one generation-phase model window.

    ``expert_ids`` has shape ``(window_token_count, num_layers,
    experts_per_token)`` and contains logical expert IDs before
    expert-parallel placement.
    """

    schema_version: int
    request_id: str
    phase: str
    window_id: int
    scheduler_step: int
    token_ids: np.ndarray
    positions: np.ndarray
    expert_ids: np.ndarray
    layer_names: tuple[str, ...]
    layer_indices: tuple[int, ...]
    window_token_count: int
    preceding_verification_token_count: int | None
    latency_ms: float
    batch_size: int
    total_batch_tokens: int
    max_sequence_length: int
    speculative_length: int


def filter_generation_rows(
    expert_ids: np.ndarray,
    token_ids: np.ndarray,
    positions: np.ndarray,
    prompt_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove prefill rows while preserving packed token order."""
    generation_mask = positions >= prompt_length
    return (
        expert_ids[generation_mask],
        token_ids[generation_mask],
        positions[generation_mask],
    )
