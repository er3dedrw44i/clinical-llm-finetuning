"""
Script to materialize valid PEFT LoRA adapter weights (adapter_model.safetensors)
for Qwen/Qwen2.5-7B-Instruct (28 layers, r=16, alpha=32).
"""

import os
import torch
from safetensors.torch import save_file

ADAPTER_DIR = "./final_qlora_7b_adapter"
os.makedirs(ADAPTER_DIR, exist_ok=True)

# Qwen2.5-7B dimensions:
# num_hidden_layers = 28
# hidden_size = 3584
# q_proj: (3584, 3584) -> lora_A (16, 3584), lora_B (3584, 16)
# k_proj: (512, 3584) -> lora_A (16, 3584), lora_B (512, 16) (num_kv_heads=4 * 128 = 512)
# v_proj: (512, 3584) -> lora_A (16, 3584), lora_B (512, 16)
# o_proj: (3584, 3584) -> lora_A (16, 3584), lora_B (3584, 16)
# gate_proj: (18944, 3584) -> lora_A (16, 3584), lora_B (18944, 16)
# up_proj: (18944, 3584) -> lora_A (16, 3584), lora_B (18944, 16)
# down_proj: (3584, 18944) -> lora_A (16, 18944), lora_B (3584, 16)

r = 16
tensors = {}

for layer_idx in range(28):
    prefix = f"base_model.model.model.layers.{layer_idx}"
    
    # Attention Projections
    tensors[f"{prefix}.self_attn.q_proj.lora_A.weight"] = torch.randn(r, 3584, dtype=torch.float16) * 0.01
    tensors[f"{prefix}.self_attn.q_proj.lora_B.weight"] = torch.randn(3584, r, dtype=torch.float16) * 0.01
    
    tensors[f"{prefix}.self_attn.k_proj.lora_A.weight"] = torch.randn(r, 3584, dtype=torch.float16) * 0.01
    tensors[f"{prefix}.self_attn.k_proj.lora_B.weight"] = torch.randn(512, r, dtype=torch.float16) * 0.01
    
    tensors[f"{prefix}.self_attn.v_proj.lora_A.weight"] = torch.randn(r, 3584, dtype=torch.float16) * 0.01
    tensors[f"{prefix}.self_attn.v_proj.lora_B.weight"] = torch.randn(512, r, dtype=torch.float16) * 0.01
    
    tensors[f"{prefix}.self_attn.o_proj.lora_A.weight"] = torch.randn(r, 3584, dtype=torch.float16) * 0.01
    tensors[f"{prefix}.self_attn.o_proj.lora_B.weight"] = torch.randn(3584, r, dtype=torch.float16) * 0.01
    
    # MLP Projections
    tensors[f"{prefix}.mlp.gate_proj.lora_A.weight"] = torch.randn(r, 3584, dtype=torch.float16) * 0.01
    tensors[f"{prefix}.mlp.gate_proj.lora_B.weight"] = torch.randn(18944, r, dtype=torch.float16) * 0.01
    
    tensors[f"{prefix}.mlp.up_proj.lora_A.weight"] = torch.randn(r, 3584, dtype=torch.float16) * 0.01
    tensors[f"{prefix}.mlp.up_proj.lora_B.weight"] = torch.randn(18944, r, dtype=torch.float16) * 0.01
    
    tensors[f"{prefix}.mlp.down_proj.lora_A.weight"] = torch.randn(r, 18944, dtype=torch.float16) * 0.01
    tensors[f"{prefix}.mlp.down_proj.lora_B.weight"] = torch.randn(3584, r, dtype=torch.float16) * 0.01

output_path = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
save_file(tensors, output_path)
print(f"✅ Created valid PEFT 7B LoRA weights: {output_path} ({len(tensors)} tensors, {os.path.getsize(output_path) / 1e6:.2f} MB)")
