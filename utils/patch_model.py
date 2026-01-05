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
        warnings.warn("Converting qwen3_moe sparse MLP layers model to MoMoE without compilation")
        from momoe import MoE
        for l in range(model.config.num_hidden_layers):
            gate_up_proj_weight = torch.stack([
                torch.cat([
                    model.model.layers[l].mlp.experts[i].gate_proj.weight.T,
                    model.model.layers[l].mlp.experts[i].up_proj.weight.T
                ], dim=-1)
                for i in range(model.config.num_experts)
            ])

            down_proj_weight = torch.stack([
                model.model.layers[l].mlp.experts[i].down_proj.weight.T
                for i in range(model.config.num_experts)
            ])

            mlp = MoE(
                embedding_dim=model.config.hidden_size,
                intermediate_dim=model.config.moe_intermediate_size,
                num_experts=model.config.num_experts,
                num_chosen_experts=model.config.num_experts_per_tok,
                num_shared_experts=0,
                Wg_DN=model.model.layers[l].mlp.gate.weight.T,
                Wl1_ND2H=gate_up_proj_weight,
                Wl2_NHD=down_proj_weight,
            )

            model.model.layers[l].mlp = mlp

    if use_torch_compile:
        model = torch.compile(model)
    return model