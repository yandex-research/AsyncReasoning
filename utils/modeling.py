"""
Utility for optimizing qwen3_moe inference
"""
import os
import warnings

import torch
import torch.nn as nn
import transformers


def prepare_model_for_inference(
        model: transformers.PreTrainedModel, *,
        use_torch_compile: bool = bool(int(os.environ.get("USE_TORCH_COMPILE", 1))),
        fuse_qwen3_moe_experts: bool = bool(int(os.environ.get("FUSE_QWEN3_MOE_EXPERTS", 1))),
        quantize_qwen3_moe_experts: bool = bool(int(os.environ.get("QUANTIZE_QWEN3_MOE_EXPERTS", 0))),
        **kwargs
) -> transformers.PreTrainedModel:
    assert not kwargs, f"unrecognized {kwargs=}"
    if model.config.model_type == "qwen3":
        pass  # no conversion - compile later
    if model.config.model_type == "llama":
        pass
    elif model.config.model_type == "qwen3_moe" and fuse_qwen3_moe_experts:
        warnings.warn("Converting qwen3_moe sparse MLP layers model to qwen3_moe_fused; full-model compile is disabled")
        use_torch_compile = False
        if quantize_qwen3_moe_experts:
            warnings.warn("Experts will be quantized to bnb 4-bit")
        transformers.utils.generic.OutputRecorder = getattr(transformers.utils.generic, "OutputRecorder", None)
        from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedSparseMoeBlock
        with torch.no_grad():
            default_device, default_dtype = torch.get_default_device(), torch.get_default_dtype()
            try:
                for i in range(len(model.model.layers)):
                    original_mlp = model.model.layers[i].mlp
                    if quantize_qwen3_moe_experts:  # CPU MoE can optionally be initialized on CPU
                        try:
                            target_device = next(p.device for p in original_mlp.parameters() if p.device.type == "cuda")
                        except StopIteration:
                            target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        original_mlp = original_mlp.to(target_device)

                    torch.set_default_device(next(original_mlp.parameters()).device)
                    torch.set_default_dtype(next(original_mlp.parameters()).dtype)
                    fused_mlp = Qwen3MoeFusedSparseMoeBlock(model.config)
                    with torch.no_grad():
                        fused_mlp.gate.weight[...] = original_mlp.gate.weight
                        assert original_mlp.gate.bias is None
                        fused_mlp.gate_proj.weight[...] = torch.stack([e.gate_proj.weight for e in original_mlp.experts])
                        fused_mlp.up_proj.weight[...] = torch.stack([e.up_proj.weight for e in original_mlp.experts])
                        fused_mlp.down_proj.weight[...] = torch.stack(
                            [e.down_proj.weight for e in original_mlp.experts])
                    if quantize_qwen3_moe_experts:
                        fused_mlp.gate_proj, fused_mlp.up_proj, fused_mlp.down_proj = map(
                            quantize_fused_linear, (fused_mlp.gate_proj, fused_mlp.up_proj, fused_mlp.down_proj))
                    model.model.layers[i].mlp = fused_mlp
                    del original_mlp, fused_mlp
            finally:
                torch.set_default_device(default_device)
                torch.set_default_dtype(default_dtype)
            model.cuda()
    elif model.config.model_type == "qwen3_moe" and not fuse_qwen3_moe_experts:
        assert not quantize_qwen3_moe_experts, "quantizing experts is currently only implemented for fused moe"
        warnings.warn("Using vanilla qwen3_moe without expert fusion / quantization")
    else:
        raise NotImplementedError(f"Unknown model type {model.config.model_type} - you can add it here")
    if use_torch_compile:
        warnings.warn("Compiling the whole model")
        model = torch.compile(model)
    return model



def quantize_fused_linear(
    fused_linear: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16, storage_dtype: torch.dtype = torch.uint8,
    quant_type: str = "nf4", compress_statistics: bool = True, blocksize: int = 64
):
    import bitsandbytes
    from qwen3_moe_fused.quantize.layer import MoeFusedLinear4bit, MoeFusedLinear
    assert isinstance(fused_linear, MoeFusedLinear)
    fused_linear_4bit = MoeFusedLinear4bit(
        in_features=fused_linear.in_features,
        out_features=fused_linear.out_features,
        num_experts=fused_linear.num_experts,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        quant_storage=storage_dtype,
        device=fused_linear.weight.device,
    )
    fused_linear_4bit.weight = bitsandbytes.nn.Params4bit(
        fused_linear.weight.data,
        requires_grad=False,
        quant_type=quant_type,
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        quant_storage=storage_dtype,
    )
    fused_linear_4bit = fused_linear_4bit.to(fused_linear.weight.device)
    fused_linear_4bit.weight.quant_state.code = fused_linear_4bit.weight.quant_state.code.float()
    return fused_linear_4bit
