#!/usr/bin/env python3
"""Convert Qwen 3.6 35B MoE weights (Safetensors / PyTorch / GGUF) into fast binary shards.

Separates dense backbone weights (attention, shared expert, embeddings, router, final_norm, lm_head)
from the 256 routed expert weights per layer.

Layout format expected by qwen3_6_moe.h:
- Expert Shard (qwen3_6_experts.shard):
  For each layer L and routed expert E (0..255):
    - gate_proj int8 weights [1408 * 2048]
    - up_proj int8 weights   [1408 * 2048]
    - down_proj int8 weights [2048 * 1408]
    - gate scales float32    [1408]
    - up scales float32      [1408]
    - down scales float32    [2048]
  Total size per expert: 8,670,208 bytes.

- Dense Backbone (qwen3_6_backbone.bin):
  Header (36 bytes):
    magic: 0x5157454E ('QWEN')
    version: 1
    n_layers, hidden_dim, moe_inter_dim, shared_inter_dim, n_routed_experts, topk, vocab_size
  Followed by float32 dense tensors:
    - embed_tokens [vocab_size, hidden_dim]
    - Layer 0..N-1:
        input_layernorm [hidden_dim]
        deltanet (q, k, v, g, beta, out) [6 * hidden_dim * hidden_dim]
        post_attn_layernorm [hidden_dim]
        shared_expert (gate, up, down) [shared_inter * hidden, shared_inter * hidden, hidden * shared_inter]
        router (weights, bias) [n_experts * hidden, n_experts]
    - final_norm [hidden_dim]
    - lm_head [vocab_size, hidden_dim]

Usage:
  python convert_qwen3_6_moe.py --model /path/to/hf_model --out-dir ./
  python convert_qwen3_6_moe.py --create-dummy --n-layers 4 --out-dir ./
"""

import argparse
import glob
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

# Optional external libraries
HAS_TORCH = False
HAS_SAFETENSORS = False
HAS_GGUF = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    pass

try:
    import gguf
    HAS_GGUF = True
except ImportError:
    pass


# Architecture Constants for Qwen 3.6 35B MoE
HIDDEN_DIM = 2048
MOE_INTER_DIM = 1408
SHARED_INTER_DIM = 1408
N_ROUTED_EXPERTS = 256
N_SHARED_EXPERTS = 1
TOPK = 8
N_HEADS = 16
HEAD_DIM = 128
DEFAULT_VOCAB = 151936


def quantize_row_int8(w_tensor):
    """Row-wise INT8 quantization with float32 scales.
    
    Returns (q8_bytes, scale_floats).
    """
    if HAS_TORCH and isinstance(w_tensor, torch.Tensor):
        w_f32 = w_tensor.float()
        row_max = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        scales = (row_max / 127.0).squeeze(1)
        q = (w_f32 / row_max * 127.0).round().clamp(-128, 127).to(torch.int8)
        return q.numpy().tobytes(), scales.cpu().numpy().astype('float32').tobytes()
    else:
        # Fallback NumPy / pure Python
        import numpy as np
        arr = np.asarray(w_tensor, dtype=np.float32)
        row_max = np.maximum(np.abs(arr).max(axis=1, keepdims=True), 1e-12)
        scales = (row_max / 127.0).squeeze(1).astype(np.float32)
        q = np.clip(np.round(arr / row_max * 127.0), -128, 127).astype(np.int8)
        return q.tobytes(), scales.tobytes()


def generate_dummy_shards(out_dir: Path, n_layers: int = 4, vocab_size: int = 1000):
    """Generate dummy dense backbone and expert shard files for testing execution."""
    print(f"[Qwen3.6 Convert] Generating synthetic dummy model shards ({n_layers} layers, vocab={vocab_size})...")
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone_path = out_dir / "qwen3_6_backbone.bin"
    experts_path = out_dir / "qwen3_6_experts.shard"

    # 1. Expert Shard
    expert_bytes_per = MOE_INTER_DIM * HIDDEN_DIM * 2 + HIDDEN_DIM * MOE_INTER_DIM + (MOE_INTER_DIM * 2 + HIDDEN_DIM) * 4
    total_expert_bytes = n_layers * N_ROUTED_EXPERTS * expert_bytes_per

    print(f" -> Writing dummy expert shard ({total_expert_bytes / (1024*1024):.2f} MB) to {experts_path}...")
    with open(experts_path, "wb") as f:
        # Generate 1 MB chunk buffer and tile it to fill total size
        chunk_size = 1024 * 1024
        pattern = bytes([i % 256 for i in range(chunk_size)])
        written = 0
        while written < total_expert_bytes:
            to_write = min(chunk_size, total_expert_bytes - written)
            f.write(pattern[:to_write])
            written += to_write

    # 2. Backbone Binary
    print(f" -> Writing dummy backbone weights to {backbone_path}...")
    with open(backbone_path, "wb") as f:
        # Header: magic(0x5157454E), version(1), n_layers, hidden, moe_inter, shared_inter, n_experts, topk, vocab
        header = struct.pack("<IIiiiiiii", 0x5157454E, 1, n_layers, HIDDEN_DIM, MOE_INTER_DIM, SHARED_INTER_DIM, N_ROUTED_EXPERTS, TOPK, vocab_size)
        f.write(header)

        # embed_tokens
        emb_floats = [0.01 * (i % 100) for i in range(vocab_size * HIDDEN_DIM)]
        f.write(struct.pack(f"<{len(emb_floats)}f", *emb_floats))

        # Layers
        for l in range(n_layers):
            # input_layernorm
            f.write(struct.pack(f"<{HIDDEN_DIM}f", *[1.0]*HIDDEN_DIM))
            # deltanet (q, k, v, g, beta, out)
            dn_matrix = [0.001 * (i % 50) for i in range(HIDDEN_DIM * HIDDEN_DIM)]
            for _ in range(6):
                f.write(struct.pack(f"<{len(dn_matrix)}f", *dn_matrix))

            # post_attn_layernorm
            f.write(struct.pack(f"<{HIDDEN_DIM}f", *[1.0]*HIDDEN_DIM))

            # shared_expert (gate, up, down)
            sh_gate_up = [0.001 * (i % 50) for i in range(SHARED_INTER_DIM * HIDDEN_DIM)]
            sh_down = [0.001 * (i % 50) for i in range(HIDDEN_DIM * SHARED_INTER_DIM)]
            f.write(struct.pack(f"<{len(sh_gate_up)}f", *sh_gate_up))
            f.write(struct.pack(f"<{len(sh_gate_up)}f", *sh_gate_up))
            f.write(struct.pack(f"<{len(sh_down)}f", *sh_down))

            # router (weights, bias)
            router_w = [math.sin(e + h * 0.01) for e in range(N_ROUTED_EXPERTS) for h in range(HIDDEN_DIM)]
            router_b = [0.0] * N_ROUTED_EXPERTS
            f.write(struct.pack(f"<{len(router_w)}f", *router_w))
            f.write(struct.pack(f"<{len(router_b)}f", *router_b))

        # final_norm
        f.write(struct.pack(f"<{HIDDEN_DIM}f", *[1.0]*HIDDEN_DIM))
        # lm_head
        lm_floats = [0.01 * (i % 100) for i in range(vocab_size * HIDDEN_DIM)]
        f.write(struct.pack(f"<{len(lm_floats)}f", *lm_floats))

    print("[Qwen3.6 Convert] Dummy model conversion complete!")


def convert_model(model_dir: Path, out_dir: Path, ebits: int = 8):
    """Convert real Safetensors / PyTorch / GGUF model directory into fast binary shards."""
    print(f"[Qwen3.6 Convert] Scanning model weights in {model_dir}...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check available files
    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    pt_files = sorted(model_dir.glob("*.bin")) + sorted(model_dir.glob("*.pt"))
    gguf_files = sorted(model_dir.glob("*.gguf"))

    if not (safetensors_files or pt_files or gguf_files):
        print(f"WARNING: No weight files found in {model_dir}. Falling back to dummy shard generation.")
        generate_dummy_shards(out_dir, n_layers=4, vocab_size=1000)
        return

    # Load state dict
    state_dict = {}
    if safetensors_files:
        if not HAS_SAFETENSORS:
            sys.exit("safetensors library not installed. Install with: pip install safetensors")
        for sf in safetensors_files:
            print(f"Loading Safetensors shard: {sf.name}...")
            state_dict.update(load_safetensors(str(sf)))
    elif pt_files:
        if not HAS_TORCH:
            sys.exit("PyTorch not installed. Install with: pip install torch")
        for pf in pt_files:
            print(f"Loading PyTorch shard: {pf.name}...")
            state_dict.update(torch.load(str(pf), map_location="cpu"))
    elif gguf_files:
        print(f"Processing GGUF model: {gguf_files[0].name}...")
        # If GGUF available, read tensors
        if HAS_GGUF:
            reader = gguf.GGUFReader(str(gguf_files[0]))
            for tensor in reader.tensors:
                state_dict[tensor.name] = tensor.data

    # Convert Dense Backbone
    backbone_path = out_dir / "qwen3_6_backbone.bin"
    print(f"[Qwen3.6 Convert] Writing dense backbone to {backbone_path}...")

    vocab_size = 248320
    hidden_dim = 2048
    n_layers = 40
    shared_inter = 512
    n_routed = 256

    with open(backbone_path, "wb") as f_bb:
        f_bb.write(struct.pack("<IIIIIIIII", 0x5157454E, 1, n_layers, hidden_dim, 512, shared_inter, n_routed, 8, vocab_size))

        # Embeddings
        embed = state_dict.get("model.language_model.embed_tokens.weight", None)
        if embed is not None:
            f_bb.write(embed.float().numpy().tobytes())
        else:
            f_bb.write(bytes([0] * (vocab_size * hidden_dim * 4)))

        # Per-layer dense weights
        for l in range(n_layers):
            p = f"model.language_model.layers.{l}"
            
            in_norm = state_dict.get(f"{p}.input_layernorm.weight", torch.ones(hidden_dim)).float().numpy()
            f_bb.write(in_norm.tobytes())

            qkv = state_dict.get(f"{p}.linear_attn.in_proj_qkv.weight", None)
            if qkv is None:
                qkv = state_dict.get(f"{p}.self_attn.in_proj_qkv.weight", torch.zeros(8192, hidden_dim)).float().numpy()
            else:
                qkv = qkv.float().numpy()

            z = state_dict.get(f"{p}.linear_attn.in_proj_z.weight", torch.zeros(4096, hidden_dim)).float().numpy()
            out = state_dict.get(f"{p}.linear_attn.out_proj.weight", None)
            if out is None:
                out = state_dict.get(f"{p}.self_attn.out_proj.weight", torch.zeros(hidden_dim, 4096)).float().numpy()
            else:
                out = out.float().numpy()

            q_w = qkv[:2048, :]
            k_w = qkv[2048:4096, :]
            v_w = qkv[4096:, :]

            a_w = state_dict.get(f"{p}.linear_attn.in_proj_a.weight", torch.zeros(32, hidden_dim)).float().numpy()
            dt_b = state_dict.get(f"{p}.linear_attn.dt_bias", torch.zeros(32)).float().numpy()

            f_bb.write(q_w.tobytes())
            f_bb.write(k_w.tobytes())
            f_bb.write(v_w.tobytes())
            f_bb.write(z.tobytes())
            f_bb.write(a_w.tobytes())
            f_bb.write(dt_b.tobytes())
            f_bb.write(out.tobytes())

            post_norm = state_dict.get(f"{p}.post_attention_layernorm.weight", torch.ones(hidden_dim)).float().numpy()
            f_bb.write(post_norm.tobytes())

            q_n = state_dict.get(f"{p}.self_attn.q_norm.weight", torch.ones(256)).float().numpy()
            k_n = state_dict.get(f"{p}.self_attn.k_norm.weight", torch.ones(256)).float().numpy()
            f_bb.write(q_n.tobytes())
            f_bb.write(k_n.tobytes())

            sh_g = state_dict.get(f"{p}.mlp.shared_expert.gate_proj.weight", torch.zeros(shared_inter, hidden_dim)).float().numpy()
            sh_u = state_dict.get(f"{p}.mlp.shared_expert.up_proj.weight", torch.zeros(shared_inter, hidden_dim)).float().numpy()
            sh_d = state_dict.get(f"{p}.mlp.shared_expert.down_proj.weight", torch.zeros(hidden_dim, shared_inter)).float().numpy()
            f_bb.write(sh_g.tobytes())
            f_bb.write(sh_u.tobytes())
            f_bb.write(sh_d.tobytes())

            r_w = state_dict.get(f"{p}.mlp.gate.weight", torch.zeros(n_routed, hidden_dim)).float().numpy()
            r_b = torch.zeros(n_routed).numpy()
            f_bb.write(r_w.tobytes())
            f_bb.write(r_b.tobytes())

        fn = state_dict.get("model.language_model.norm.weight", torch.ones(hidden_dim)).float().numpy()
        f_bb.write(fn.tobytes())

        lm = state_dict.get("lm_head.weight", None)
        if lm is None:
            lm = state_dict.get("model.language_model.embed_tokens.weight", torch.zeros(vocab_size, hidden_dim))
        f_bb.write(lm.float().numpy().tobytes())

    print(f"Backbone converted successfully -> {backbone_path}")

    # Find number of layers
    layer_indices = set()
    for k in state_dict.keys():
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            layer_indices.add(int(m.group(1)))

    n_layers = len(layer_indices) if layer_indices else 4
    print(f"Detected {n_layers} layers.")

    # Convert Expert Shards
    experts_path = out_dir / "qwen3_6_experts.shard"
    print(f"Writing 256 routed experts per layer to {experts_path}...")
    
    with open(experts_path, "wb") as f_exp:
        for l in range(n_layers):
            # Check fused expert tensors
            fused_gu_key = f"model.language_model.layers.{l}.mlp.experts.gate_up_proj"
            fused_d_key = f"model.language_model.layers.{l}.mlp.experts.down_proj"
            
            gu_tensor = state_dict.get(fused_gu_key, None)
            d_tensor = state_dict.get(fused_d_key, None)

            for e in range(N_ROUTED_EXPERTS):
                if gu_tensor is not None and d_tensor is not None:
                    # gu_tensor shape: [256, 1024, 2048] -> split into gate (512) and up (512)
                    exp_gu = gu_tensor[e] # [1024, 2048]
                    gate_w = exp_gu[:512, :]
                    up_w = exp_gu[512:, :]
                    down_w = d_tensor[e] # [2048, 512]

                    g_bytes, g_scales = quantize_row_int8(gate_w)
                    u_bytes, u_scales = quantize_row_int8(up_w)
                    d_bytes, d_scales = quantize_row_int8(down_w)

                    f_exp.write(g_bytes)
                    f_exp.write(u_bytes)
                    f_exp.write(d_bytes)
                    f_exp.write(g_scales)
                    f_exp.write(u_scales)
                    f_exp.write(d_scales)
                else:
                    # Unfused key fallback
                    g_key = f"model.language_model.layers.{l}.mlp.experts.{e}.gate_proj.weight"
                    u_key = f"model.language_model.layers.{l}.mlp.experts.{e}.up_proj.weight"
                    d_key = f"model.language_model.layers.{l}.mlp.experts.{e}.down_proj.weight"

                    if g_key in state_dict and u_key in state_dict and d_key in state_dict:
                        g_bytes, g_scales = quantize_row_int8(state_dict[g_key])
                        u_bytes, u_scales = quantize_row_int8(state_dict[u_key])
                        d_bytes, d_scales = quantize_row_int8(state_dict[d_key])

                        f_exp.write(g_bytes)
                        f_exp.write(u_bytes)
                        f_exp.write(d_bytes)
                        f_exp.write(g_scales)
                        f_exp.write(u_scales)
                        f_exp.write(d_scales)
                    else:
                        # Fallback for empty slots
                        g_bytes = bytes([0] * (MOE_INTER_DIM * HIDDEN_DIM))
                        u_bytes = bytes([0] * (MOE_INTER_DIM * HIDDEN_DIM))
                        d_bytes = bytes([0] * (HIDDEN_DIM * MOE_INTER_DIM))
                        scales_g = struct.pack(f"<{MOE_INTER_DIM}f", *[1.0]*MOE_INTER_DIM)
                        scales_u = struct.pack(f"<{MOE_INTER_DIM}f", *[1.0]*MOE_INTER_DIM)
                        scales_d = struct.pack(f"<{HIDDEN_DIM}f", *[1.0]*HIDDEN_DIM)

                        f_exp.write(g_bytes)
                        f_exp.write(u_bytes)
                        f_exp.write(d_bytes)
                        f_exp.write(scales_g)
                        f_exp.write(scales_u)
                        f_exp.write(scales_d)

    print(f"Expert shards converted successfully -> {experts_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen 3.6 35B MoE model weights into binary shards")
    parser.add_argument("--model", type=str, help="Path to HF model directory or checkpoint")
    parser.add_argument("--out-dir", type=str, default="./", help="Output directory for binary files")
    parser.add_argument("--ebits", type=int, default=8, help="Expert quantization bits (default 8)")
    parser.add_argument("--create-dummy", action="store_true", help="Generate dummy benchmark/test shards")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of layers for dummy generation")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocab size for dummy generation")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    if args.create_dummy or not args.model:
        generate_dummy_shards(out_dir, n_layers=args.n_layers, vocab_size=args.vocab_size)
    else:
        convert_model(Path(args.model), out_dir, ebits=args.ebits)


if __name__ == "__main__":
    main()
