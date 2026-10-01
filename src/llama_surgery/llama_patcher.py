import torch.nn as nn
from .surgery import SurgicalLlamaAttention

def patch_llama_model(
    model,
    tree_depth: int = 4,
    arity: int = 2,
    tau_init: float = 1.0,
    preserve_sinks: bool = True,
    req_depth: int = None,
    max_dist: int = None,
    **kwargs
):
    """
    Surgically injects Dynamic Topology Routers into a pre-trained LLaMA model,
    replacing self_attn layers with SurgicalLlamaAttention and transferring pre-trained weights.
    """
    setattr(model.config, "surgical_p", arity)
    setattr(model.config, "surgical_tree_depth", tree_depth)
    setattr(model.config, "surgical_tau_init", tau_init)
    setattr(model.config, "preserve_sinks", preserve_sinks)
    if req_depth is not None:
        setattr(model.config, "surgical_req_depth", req_depth)
    if max_dist is not None:
        setattr(model.config, "surgical_max_dist", max_dist)

    for i, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        
        # Instantiate new attention
        new_attn = SurgicalLlamaAttention(model.config, layer_idx=i)
        
        # Copy weights
        new_attn.q_proj.weight = old_attn.q_proj.weight
        new_attn.k_proj.weight = old_attn.k_proj.weight
        new_attn.v_proj.weight = old_attn.v_proj.weight
        new_attn.o_proj.weight = old_attn.o_proj.weight
        
        # If bias exists, copy that too
        if hasattr(old_attn.q_proj, 'bias') and old_attn.q_proj.bias is not None:
            new_attn.q_proj.bias = old_attn.q_proj.bias
        if hasattr(old_attn.k_proj, 'bias') and old_attn.k_proj.bias is not None:
            new_attn.k_proj.bias = old_attn.k_proj.bias
        if hasattr(old_attn.v_proj, 'bias') and old_attn.v_proj.bias is not None:
            new_attn.v_proj.bias = old_attn.v_proj.bias
        if hasattr(old_attn.o_proj, 'bias') and old_attn.o_proj.bias is not None:
            new_attn.o_proj.bias = old_attn.o_proj.bias

        # Move to the same device and dtype
        new_attn.to(old_attn.q_proj.weight.device, dtype=old_attn.q_proj.weight.dtype)
        
        # Replace the module
        layer.self_attn = new_attn
    
    return model

def inject_surgery(model, **kwargs):
    """
    Iterates over model.model.layers, replaces self_attn with SurgicalLlamaAttention,
    and copies over the pre-trained q_proj, k_proj, v_proj, o_proj weights.
    """
    return patch_llama_model(model, **kwargs)
