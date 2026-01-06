"""
Utility for optimizing qwen3_moe inference
"""
import os
import warnings

import torch
import transformers


def prepare_model_for_inference(
        model: transformers.PreTrainedModel, *,
        use_torch_compile: bool = bool(int(os.environ.get("USE_TORCH_COMPILE", 1))),
        **kwargs
) -> transformers.PreTrainedModel:
    assert not kwargs, f"unrecognized {kwargs=}"
    if model.config.model_type == "qwen3":
        pass  # no conversion - compile later
    elif model.config.model_type == "qwen3_moe":
        warnings.warn("Converting qwen3_moe sparse MLP layers model to qwen3_moe_fused")
        transformers.utils.generic.OutputRecorder = getattr(transformers.utils.generic, "OutputRecorder", None)
        from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedSparseMoeBlock
        with torch.no_grad():
            default_device, default_dtype = torch.get_default_device(), torch.get_default_dtype()
            try:
                for i in range(len(model.model.layers)):
                    original_mlp = model.model.layers[i].mlp
                    torch.set_default_device(next(original_mlp.parameters()).device)
                    torch.set_default_dtype(next(original_mlp.parameters()).dtype)
                    fused_mlp = Qwen3MoeFusedSparseMoeBlock(model.config)
                    with torch.no_grad():
                        fused_mlp.gate.weight[...] = original_mlp.gate.weight
                        assert original_mlp.gate.bias is None
                        fused_mlp.gate_proj.weight[...] = torch.stack(
                            [e.gate_proj.weight for e in original_mlp.experts])
                        fused_mlp.up_proj.weight[...] = torch.stack([e.up_proj.weight for e in original_mlp.experts])
                        fused_mlp.down_proj.weight[...] = torch.stack(
                            [e.down_proj.weight for e in original_mlp.experts])
                    model.model.layers[i].mlp = fused_mlp
                    del original_mlp, fused_mlp
            finally:
                torch.set_default_device(default_device)
                torch.set_default_dtype(default_dtype)
    else:
        raise NotImplementedError(f"Unknown model type {model.config.model_type} - you can add it here")
    if use_torch_compile:
        model = torch.compile(model)
    return model