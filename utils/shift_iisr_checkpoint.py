from pathlib import Path
from typing import Mapping, Optional

import torch


FORMAT_VERSION = 1
MODEL_NAME = "Shift-IISR"
GRM_SCHEDULE = [1.0, 0.67, 0.33, 0.0]
LSR_SCHEDULE = [0.0, 0.2, 0.7, 1.0]


def _validate_state_dict(state_dict, name):
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"{name} must be a non-empty state dict.")
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise ValueError(f"{name} contains non-tensor values.")


def build_shift_iisr_checkpoint(
    grm_feature_extractor_state,
    grm_projector_state,
    checkpoint_step: int,
    lsr_strength: float = 1.0,
    base_unet: str = "resshift_bicsrx4_s4.pth",
    autoencoder: str = "autoencoder_vq_f4.pth",
    extra_metadata: Optional[dict] = None,
):
    _validate_state_dict(grm_feature_extractor_state, "grm_feature_extractor")
    _validate_state_dict(grm_projector_state, "grm_projector")

    metadata = {
        "model_name": MODEL_NAME,
        "checkpoint_type": "inference",
        "checkpoint_step": int(checkpoint_step),
        "contains_classifier": False,
        "contains_base_models": False,
        "grm_schedule": GRM_SCHEDULE,
        "lsr_schedule": LSR_SCHEDULE,
        "lsr_strength": float(lsr_strength),
        "base_unet": base_unet,
        "autoencoder": autoencoder,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "format_version": FORMAT_VERSION,
        "model_name": MODEL_NAME,
        "checkpoint_step": int(checkpoint_step),
        "grm_feature_extractor": grm_feature_extractor_state,
        "grm_projector": grm_projector_state,
        "metadata": metadata,
    }


def save_shift_iisr_checkpoint(
    output_path,
    grm_feature_extractor,
    grm_projector,
    checkpoint_step: int,
    lsr_strength: float = 1.0,
):
    checkpoint = build_shift_iisr_checkpoint(
        grm_feature_extractor.state_dict(),
        grm_projector.state_dict(),
        checkpoint_step=checkpoint_step,
        lsr_strength=lsr_strength,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return output_path


def load_shift_iisr_checkpoint(
    checkpoint_path,
    grm_feature_extractor,
    grm_projector,
    map_location="cpu",
    strict=True,
    expected_lsr_strength=None,
):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Shift-IISR checkpoint must be a dictionary.")
    if checkpoint.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"Unexpected model_name: {checkpoint.get('model_name')!r}; "
            f"expected {MODEL_NAME!r}."
        )
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported Shift-IISR checkpoint format: "
            f"{checkpoint.get('format_version')!r}."
        )

    metadata = checkpoint.get("metadata", {})
    if metadata.get("grm_schedule") != GRM_SCHEDULE:
        raise ValueError("Shift-IISR checkpoint has an unexpected GRM schedule.")
    if metadata.get("lsr_schedule") != LSR_SCHEDULE:
        raise ValueError("Shift-IISR checkpoint has an unexpected LSR schedule.")
    if expected_lsr_strength is not None:
        checkpoint_strength = float(metadata.get("lsr_strength"))
        if checkpoint_strength != float(expected_lsr_strength):
            raise ValueError(
                f"LSR strength mismatch: checkpoint={checkpoint_strength}, "
                f"config={float(expected_lsr_strength)}."
            )

    extractor_state = checkpoint.get("grm_feature_extractor")
    projector_state = checkpoint.get("grm_projector")
    _validate_state_dict(extractor_state, "grm_feature_extractor")
    _validate_state_dict(projector_state, "grm_projector")

    grm_feature_extractor.load_state_dict(extractor_state, strict=strict)
    grm_projector.load_state_dict(projector_state, strict=strict)
    return checkpoint
