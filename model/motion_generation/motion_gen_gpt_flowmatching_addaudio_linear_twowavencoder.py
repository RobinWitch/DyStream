import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import PretrainedConfig
from timm.models.vision_transformer import Mlp
from transformers import PreTrainedModel, Wav2Vec2Processor, Wav2Vec2Model, PretrainedConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Attention
from typing import Optional, Tuple, Union
from .motion_gen_utils_dev import WanTimeEmbedding
import time


class RoPEEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)
        theta = 10000 ** (-2 * torch.arange(0, d_model // 2, dtype=torch.float) / d_model)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        angles = positions * theta
        self.register_buffer('cos_angles', angles.cos())
        self.register_buffer('sin_angles', angles.sin())

    def forward(self, x, offset=0):
        seq_len = x.size(1)
        cos_angles = self.cos_angles[offset:offset + seq_len]
        sin_angles = self.sin_angles[offset:offset + seq_len]
        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]
        x_rot = x.clone()
        x_rot[:, :, 0::2] = x_even * cos_angles - x_odd * sin_angles
        x_rot[:, :, 1::2] = x_even * sin_angles + x_odd * cos_angles
        return self.dropout(x_rot)


class SelfAttention_Rope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len=128):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rope = RoPEEncoding(d_model, dropout=dropout, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        x = self.self_attn(
            self.rope(x),
            self.rope(x),
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x


class CrossAttention_Rope(nn.Module):
    def __init__(self, d_model: int, d_cond: int, num_heads: int, dropout: float = 0.1, max_seq_len=128):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=d_cond,
            vdim=d_cond,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rope = RoPEEncoding(d_model, dropout=dropout, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        x = self.cross_attn(
            self.rope(x),
            self.rope(cond),
            cond,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x


class SelfAttention_Pos(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len=128):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.pe = PositionalEncoding(
            d_model, dropout=dropout, max_seq_len=max_seq_len
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        x = self.self_attn(
            self.pe(x),
            self.pe(x),
            self.pe(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        x = x + self.pe[:, offset:offset + x.size(1), :]
        return self.dropout(x)


def _mha_with_cache(mha, q_input, k_input, v_input, kv_cache=None, is_causal=False):
    """Run nn.MultiheadAttention with optional KV-cache, bypassing its forward.

    Uses fused QKV projection when inputs share the same tensor (self-attention).

    Args:
        mha: nn.MultiheadAttention (must have _qkv_same_embed_dim=True)
        q_input, k_input, v_input: (B, L, D) inputs (pre-RoPE/PE applied)
        kv_cache: optional (cached_k, cached_v) each (B, nheads, L_cached, head_dim)
        is_causal: apply causal mask (only when kv_cache is None)

    Returns:
        output: (B, L_q, D)
        new_cache: (k, v) each (B, nheads, L_total, head_dim)
    """
    embed_dim = mha.embed_dim
    num_heads = mha.num_heads
    head_dim = embed_dim // num_heads
    B = q_input.size(0)
    w = mha.in_proj_weight
    b = mha.in_proj_bias

    # Fused QKV projection when all inputs are the same tensor
    if q_input is k_input and k_input is v_input:
        qkv = F.linear(q_input, w, b)
        q, k_new, v_new = qkv.chunk(3, dim=-1)
    elif q_input is k_input:
        # Q and K share input (self-attn with RoPE: q=k=rope(x), v=x)
        qk = F.linear(q_input, w[:2*embed_dim], b[:2*embed_dim] if b is not None else None)
        q, k_new = qk.chunk(2, dim=-1)
        v_new = F.linear(v_input, w[2*embed_dim:], b[2*embed_dim:] if b is not None else None)
    else:
        q = F.linear(q_input, w[:embed_dim], b[:embed_dim] if b is not None else None)
        k_new = F.linear(k_input, w[embed_dim:2*embed_dim], b[embed_dim:2*embed_dim] if b is not None else None)
        v_new = F.linear(v_input, w[2*embed_dim:], b[2*embed_dim:] if b is not None else None)

    q = q.view(B, -1, num_heads, head_dim).transpose(1, 2)
    k_new = k_new.view(B, -1, num_heads, head_dim).transpose(1, 2)
    v_new = v_new.view(B, -1, num_heads, head_dim).transpose(1, 2)

    if kv_cache is not None:
        cached_k, cached_v = kv_cache
        k = torch.cat([cached_k, k_new], dim=2)
        v = torch.cat([cached_v, v_new], dim=2)
        is_causal = False  # single query attends to all cached + new keys
    else:
        k, v = k_new, v_new

    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    attn_out = attn_out.transpose(1, 2).reshape(B, -1, embed_dim)
    out = mha.out_proj(attn_out)
    return out, (k, v)


def _block_forward_full(block, x, audio_features, anchor_hidden):
    """Audio2FaceGPTBlock full forward that captures KV-cache."""
    cache = {}

    residual = x
    x_norm = block.norm1(x)
    x_rope = block.self_attn_rope.rope(x_norm)
    attn_out, cache['sa_rope'] = _mha_with_cache(
        block.self_attn_rope.self_attn, x_rope, x_rope, x_norm, is_causal=True)
    x = residual + attn_out

    residual = x
    x = residual + block.cross_linear_audio(audio_features)

    residual = x
    x_norm = block.norm_anchor(x)
    q_rope = block.cross_attn_anchor.rope(x_norm)
    k_rope = block.cross_attn_anchor.rope(anchor_hidden)
    attn_out, _ = _mha_with_cache(
        block.cross_attn_anchor.cross_attn, q_rope, k_rope, anchor_hidden)
    x = residual + attn_out

    residual = x
    x_norm = block.norm3(x)
    x_pe = block.self_attn_pos.pe(x_norm)
    attn_out, cache['sa_pos'] = _mha_with_cache(
        block.self_attn_pos.self_attn, x_pe, x_pe, x_pe, is_causal=True)
    x = residual + attn_out

    residual = x
    x = block.norm4(x)
    x = block.mlp(x)
    x = residual + x

    return x, cache


def _block_forward_incr(block, x_new, audio_new, anchor_hidden, cache, offset):
    """Audio2FaceGPTBlock incremental forward using KV-cache. x_new is (B, 1, D)."""
    residual = x_new
    x_norm = block.norm1(x_new)
    x_rope = block.self_attn_rope.rope(x_norm, offset=offset)
    attn_out, cache['sa_rope'] = _mha_with_cache(
        block.self_attn_rope.self_attn, x_rope, x_rope, x_norm,
        kv_cache=cache['sa_rope'])
    x = residual + attn_out

    residual = x
    x = residual + block.cross_linear_audio(audio_new)

    residual = x
    x_norm = block.norm_anchor(x)
    q_rope = block.cross_attn_anchor.rope(x_norm, offset=offset)
    k_rope = block.cross_attn_anchor.rope(anchor_hidden)
    attn_out, _ = _mha_with_cache(
        block.cross_attn_anchor.cross_attn, q_rope, k_rope, anchor_hidden)
    x = residual + attn_out

    residual = x
    x_norm = block.norm3(x)
    x_pe = block.self_attn_pos.pe(x_norm, offset=offset)
    attn_out, cache['sa_pos'] = _mha_with_cache(
        block.self_attn_pos.self_attn, x_pe, x_pe, x_pe,
        kv_cache=cache['sa_pos'])
    x = residual + attn_out

    residual = x
    x = block.norm4(x)
    x = block.mlp(x)
    x = residual + x

    return x, cache


# ── Compilable GPT loop (dict-free, flat cache for torch.compile) ────────

def _gpt_blocks_full(blocks, x, audio, anchor):
    """All blocks full forward. Returns x and 4 flat cache lists (no dicts).
    Designed for torch.compile: no dict ops, no data-dependent control flow."""
    srk = []  # sa_rope K per block
    srv = []  # sa_rope V per block
    spk = []  # sa_pos K per block
    spv = []  # sa_pos V per block

    for i in range(len(blocks)):
        block = blocks[i]

        residual = x
        x_norm = block.norm1(x)
        x_rope = block.self_attn_rope.rope(x_norm)
        attn_out, sa_rope_kv = _mha_with_cache(
            block.self_attn_rope.self_attn, x_rope, x_rope, x_norm, is_causal=True)
        x = residual + attn_out

        x = x + block.cross_linear_audio(audio)

        residual = x
        x_norm = block.norm_anchor(x)
        q_rope = block.cross_attn_anchor.rope(x_norm)
        k_rope = block.cross_attn_anchor.rope(anchor)
        attn_out, _ = _mha_with_cache(
            block.cross_attn_anchor.cross_attn, q_rope, k_rope, anchor)
        x = residual + attn_out

        residual = x
        x_norm = block.norm3(x)
        x_pe = block.self_attn_pos.pe(x_norm)
        attn_out, sa_pos_kv = _mha_with_cache(
            block.self_attn_pos.self_attn, x_pe, x_pe, x_pe, is_causal=True)
        x = residual + attn_out

        residual = x
        x = block.norm4(x)
        x = block.mlp(x)
        x = residual + x

        srk.append(sa_rope_kv[0])
        srv.append(sa_rope_kv[1])
        spk.append(sa_pos_kv[0])
        spv.append(sa_pos_kv[1])

    return x, srk, srv, spk, spv


def _gpt_blocks_incr(blocks, x_new, audio_new, anchor,
                     srk, srv, spk, spv, offset):
    """All blocks incremental forward with flat cache lists."""
    new_srk = []
    new_srv = []
    new_spk = []
    new_spv = []

    for i in range(len(blocks)):
        block = blocks[i]

        residual = x_new
        x_norm = block.norm1(x_new)
        x_rope = block.self_attn_rope.rope(x_norm, offset=offset)
        attn_out, sa_rope_kv = _mha_with_cache(
            block.self_attn_rope.self_attn, x_rope, x_rope, x_norm,
            kv_cache=(srk[i], srv[i]))
        x_new = residual + attn_out

        x_new = x_new + block.cross_linear_audio(audio_new)

        residual = x_new
        x_norm = block.norm_anchor(x_new)
        q_rope = block.cross_attn_anchor.rope(x_norm, offset=offset)
        k_rope = block.cross_attn_anchor.rope(anchor)
        attn_out, _ = _mha_with_cache(
            block.cross_attn_anchor.cross_attn, q_rope, k_rope, anchor)
        x_new = residual + attn_out

        residual = x_new
        x_norm = block.norm3(x_new)
        x_pe = block.self_attn_pos.pe(x_norm, offset=offset)
        attn_out, sa_pos_kv = _mha_with_cache(
            block.self_attn_pos.self_attn, x_pe, x_pe, x_pe,
            kv_cache=(spk[i], spv[i]))
        x_new = residual + attn_out

        residual = x_new
        x_new = block.norm4(x_new)
        x_new = block.mlp(x_new)
        x_new = residual + x_new

        new_srk.append(sa_rope_kv[0])
        new_srv.append(sa_rope_kv[1])
        new_spk.append(sa_pos_kv[0])
        new_spv.append(sa_pos_kv[1])

    return x_new, new_srk, new_srv, new_spk, new_spv


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class AutoModelConfig(PretrainedConfig):
    def __init__(self, config_obj=None, **kwargs):
        if config_obj is not None:
            cfg_dict = OmegaConf.to_container(config_obj, resolve=True)
            kwargs.update(cfg_dict)
            self.model_type = kwargs.pop("model_type", "my_model")
        super().__init__(**kwargs)


class Audio2FaceGPTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)
        self.norm_anchor = nn.LayerNorm(hidden_size)
        self.self_attn_rope = SelfAttention_Rope(hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        self.cross_attn_rope = CrossAttention_Rope(hidden_size, hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        self.self_attn_pos = SelfAttention_Pos(hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        self.cross_attn_anchor = CrossAttention_Rope(hidden_size, hidden_size, num_heads, dropout, max_seq_len=max_seq_len)
        self.cross_linear_audio = nn.Linear(hidden_size, hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

    def forward(self, x, audio_features, anchor_hidden, causal_mask=None, cross_causal_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attn_rope(x, attn_mask=causal_mask)
        x = residual + x

        residual = x
        x = residual + self.cross_linear_audio(audio_features)

        residual = x
        x = self.norm_anchor(x)
        x = self.cross_attn_anchor(x, anchor_hidden, attn_mask=None)
        x = residual + x

        residual = x
        x = self.norm3(x)
        x = self.self_attn_pos(x, attn_mask=causal_mask)
        x = residual + x

        residual = x
        x = self.norm4(x)
        x = self.mlp(x)
        x = residual + x

        return x


def make_attention_causal(attn: Wav2Vec2Attention):
    q_proj, k_proj, v_proj, out_proj = attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj
    n_head, head_dim, p = attn.num_heads, attn.head_dim, attn.dropout

    def f(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **_,
    ):
        B, T, _ = x.shape
        q = q_proj(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = k_proj(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = v_proj(x).view(B, T, n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=p if self.training else 0.0,
            is_causal=False,
        )
        y = out_proj(y.transpose(1, 2).reshape(B, T, n_head * head_dim))
        return (y, None, None) if output_attentions else (y, None, None)

    attn.forward = f.__get__(attn, attn.__class__)


class WrapedWav2Vec(nn.Module):
    def __init__(self, layers: int = 1):
        super().__init__()
        base = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.feature_extractor = base.feature_extractor
        self.feature_projection = base.feature_projection
        self.encoder = base.encoder
        self.encoder.layers = self.encoder.layers[:layers]
        for l in self.encoder.layers:
            make_attention_causal(l.attention)

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **_,
    ):
        low = self.feature_extractor(x).transpose(1, 2)
        h, _ = self.feature_projection(low.detach())
        enc = self.encoder(
            h,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return {"low_level": low, "high_level": enc[0]}


class DiffusionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp1 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        self.mlp2 = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)
        self.adaLN_modulation1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        self.adaLN_modulation2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )

    def forward(self, hidden_states, gpt_hidden, temb=None):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation1(temb).chunk(3, dim=-1)
        shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN_modulation2(gpt_hidden).chunk(3, dim=-1)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.mlp1(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        norm_hidden_states = (self.norm2(hidden_states.float()) * (1 + scale_mlp2) + shift_mlp2).type_as(hidden_states)
        ff_output = self.mlp2(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp2).type_as(hidden_states)

        return hidden_states


class DiffusionHead(nn.Module):
    def __init__(
        self,
        face_dim=512,
        hidden_size=768,
        num_layers=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        max_seq_len=128,
    ):
        super().__init__()
        self.face_dim = face_dim
        self.hidden_size = hidden_size
        self.noisy_proj = nn.Linear(face_dim, hidden_size)
        self.gpt_proj = nn.Linear(face_dim, hidden_size)
        self.past_proj = nn.Linear(face_dim, hidden_size)
        self.anchor_proj = nn.Linear(face_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiffusionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, face_dim)

    def forward(self, noisy_face_latent, gpt_output, temb=None):
        bs = noisy_face_latent.shape[0]
        device = noisy_face_latent.device
        noisy_hidden = self.noisy_proj(noisy_face_latent)
        gpt_hidden = self.gpt_proj(gpt_output)
        x = noisy_hidden
        for block in self.blocks:
            x = block(x, gpt_hidden, temb)
        x = self.output_norm(x)
        denoised = self.output_proj(x)
        return denoised


class Audio2FaceGPT(nn.Module):
    def __init__(
        self,
        cfg=None,
        audio_dim=768,
        face_dim=512,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        max_seq_len=1024,
    ):
        super().__init__()
        self.cfg = cfg
        self.audio_encoder_face = WrapedWav2Vec(layers=self.cfg.wav2vec_layer)
        self.audio_encoder_face_other = WrapedWav2Vec(layers=self.cfg.wav2vec_layer)
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_dim = audio_dim
        self.face_dim = face_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.audio_proj = nn.Linear(audio_dim, hidden_size)
        self.audio_other_proj = nn.Linear(audio_dim, hidden_size)
        self.audio_audioother_fusion = nn.Linear(2 * hidden_size, hidden_size)
        self.face_embed = nn.Linear(face_dim, hidden_size)
        self.anchor_embed = nn.Linear(face_dim, hidden_size)
        self.time_embed = WanTimeEmbedding(
            dim=hidden_size,
            time_freq_dim=hidden_size,
        )
        self.blocks = nn.ModuleList(
            [
                Audio2FaceGPTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, face_dim)
        self.inpainting_length = cfg.cbh_window_length - 2
        self.diffusion_head = DiffusionHead(
            face_dim=face_dim,
            hidden_size=hidden_size,
            num_layers=6,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.cfg_all = cfg.cfg_all
        self.drop_gpt = cfg.drop_gpt
        self.cfg_audio = cfg.cfg_audio
        self.drop_audio = cfg.drop_audio
        self.cfg_audio_other = cfg.cfg_audio_other
        self.drop_audio_other = cfg.drop_audio_other
        self.cfg_anchor = cfg.cfg_anchor
        self.drop_anchor = cfg.drop_anchor
        self.cfg_audio_anchor = cfg.cfg_audio_anchor

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def generate_cross_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def get_audio2face_fea(self, audio, prev_audio, n):
        if prev_audio is not None:
            audio = torch.cat([prev_audio, audio], dim=1)
        audio_list = [i.cpu().numpy() for i in audio]
        inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio2face_fea = self.audio_encoder_face(inputs.input_values)["high_level"]
        audio2face_fea = F.interpolate(
            audio2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps / 50), mode="linear", align_corners=True
        ).transpose(1, 2)
        if prev_audio is not None:
            audio2face_fea = audio2face_fea[:, -n:]
        else:
            if audio2face_fea.shape[1] > n:
                audio2face_fea = audio2face_fea[:, :n]
            if audio2face_fea.shape[1] < n:
                audio2face_fea = torch.cat(
                    [audio2face_fea, audio2face_fea[:, -(n - audio2face_fea.shape[1]) :]],
                    dim=1,
                )
        return audio2face_fea

    def get_audio2face_fea_other(self, audio, prev_audio, n):
        if prev_audio is not None:
            audio = torch.cat([prev_audio, audio], dim=1)
        audio_list = [i.cpu().numpy() for i in audio]
        inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio2face_fea = self.audio_encoder_face_other(inputs.input_values)["high_level"]
        audio2face_fea = F.interpolate(
            audio2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps / 50), mode="linear", align_corners=True
        ).transpose(1, 2)
        if prev_audio is not None:
            audio2face_fea = audio2face_fea[:, -n:]
        else:
            if audio2face_fea.shape[1] > n:
                audio2face_fea = audio2face_fea[:, :n]
            if audio2face_fea.shape[1] < n:
                audio2face_fea = torch.cat(
                    [audio2face_fea, audio2face_fea[:, -(n - audio2face_fea.shape[1]) :]],
                    dim=1,
                )
        return audio2face_fea

    def forward(self, face_latent_gt, noise_face_latent, time_step, audio, audio_other, prev_audio, prev_audio_other, anchor_latent):
        bs, n, _ = face_latent_gt.shape
        audio2face_fea = self.get_audio2face_fea(audio, prev_audio, n)
        audio2face_fea_other = self.get_audio2face_fea_other(audio_other, prev_audio_other, n)
        device = audio2face_fea.device
        bs, seq_len, _ = audio2face_fea.shape
        audio_hidden = self.audio_proj(audio2face_fea)
        audio_hidden = audio_hidden[:, 1:]
        drop_audio_mask = torch.rand(bs, seq_len - 1, 1, device=face_latent_gt.device) < self.drop_audio
        drop_audio_mask = drop_audio_mask.float()
        audio_hidden = audio_hidden * (1 - drop_audio_mask)
        audio_other_hidden = self.audio_other_proj(audio2face_fea_other)
        audio_other_hidden = audio_other_hidden[:, 1:]
        drop_audio_other_mask = torch.rand(bs, seq_len - 1, 1, device=face_latent_gt.device) < self.drop_audio_other
        drop_audio_other_mask = drop_audio_other_mask.float()
        audio_other_hidden = audio_other_hidden * (1 - drop_audio_other_mask)
        audio_hidden = self.audio_audioother_fusion(torch.cat([audio_hidden, audio_other_hidden], dim=-1))
        anchor_hidden = self.anchor_embed(anchor_latent)
        causal_mask = self.generate_causal_mask(seq_len - 1, device)
        cross_causal_mask = self.generate_cross_causal_mask(seq_len - 1, device)
        face_hidden = self.face_embed(face_latent_gt[:, :-1])
        drop_anchor_mask = torch.rand(bs, 1, 1, device=face_hidden.device) < self.drop_anchor
        drop_anchor_mask = drop_anchor_mask.float()
        x = face_hidden
        for block in self.blocks:
            x = block(
                x,
                audio_hidden,
                anchor_hidden * (1 - drop_anchor_mask),
                causal_mask,
                cross_causal_mask,
            )
        x = self.output_norm(x)
        gpt_output = self.output_proj(x)
        time_embedding = self.time_embed(time_step).unsqueeze(1)
        output = self.diffusion_head(
            noise_face_latent[:, 1:],
            gpt_output,
            temb=time_embedding if time_step is not None else None,
        )
        return output

    def one_clip_only_inference(
        self,
        per_compute_audio_feature,
        audio_self,
        past_audio_self,
        anchor_latent,
        past_motion,
        gen_frames,
        per_compute_audio_other_feature=None,
        audio_other=None,
        past_audio_other=None,
        noise_scheduler=None,
        num_inference_steps=10,
    ):
        use_pre_compute_audio_feature = (
            per_compute_audio_feature is not None and per_compute_audio_other_feature is not None
        )
        audio = audio_self
        n = gen_frames + self.inpainting_length + 1
        audio2face_fea = None
        audio2face_fea_other = None
        if not use_pre_compute_audio_feature:
            audio2face_fea = self.get_audio2face_fea(audio_self, past_audio_self, n)
            audio2face_fea_other = self.get_audio2face_fea_other(audio_other, past_audio_other, n)
        device = audio.device
        if noise_scheduler is not None:
            noise_scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = noise_scheduler.timesteps
        audio_features = per_compute_audio_feature if use_pre_compute_audio_feature else audio2face_fea
        audio_other_features = per_compute_audio_other_feature if use_pre_compute_audio_feature else audio2face_fea_other
        audio_features = audio_features[:, 1:]
        audio_other_features = audio_other_features[:, 1:]
        bs, seq_len, _ = audio_features.shape
        device = audio_features.device
        audio_hidden = self.audio_proj(audio_features)
        audio_other_hidden = self.audio_other_proj(audio_other_features)
        audio_zeros = torch.zeros_like(audio_hidden)
        audio_other_zeros = torch.zeros_like(audio_other_hidden)
        audio_hidden_0 = self.audio_audioother_fusion(torch.cat([audio_zeros, audio_other_zeros], dim=-1))
        audio_hidden_1 = self.audio_audioother_fusion(torch.cat([audio_hidden, audio_other_zeros], dim=-1))
        audio_hidden_2 = self.audio_audioother_fusion(torch.cat([audio_zeros, audio_other_hidden], dim=-1))
        audio_hidden_3 = self.audio_audioother_fusion(torch.cat([audio_hidden, audio_other_hidden], dim=-1))
        anchor_hidden = self.anchor_embed(anchor_latent)
        causal_mask = self.generate_causal_mask(seq_len, device)
        cross_causal_mask = self.generate_cross_causal_mask(seq_len, device)
        face_hidden_last = self.face_embed(past_motion)
        face_hidden = torch.zeros(bs, seq_len, self.hidden_size, device=device)
        face_hidden[:, :self.inpainting_length] = face_hidden_last
        # Pre-compute anchor_hidden_input (constant across loop iterations)
        anchor_zeros = torch.zeros_like(anchor_hidden)
        skip_anchor = (self.cfg_anchor == 0.0)
        n_cfg = 4 if skip_anchor else 5
        if skip_anchor:
            # 4 patterns: unconditional, audio_self, audio_other, all
            anchor_hidden_input = torch.cat(
                [anchor_zeros, anchor_zeros, anchor_zeros, anchor_hidden],
                dim=0,
            )
            audio_hidden_stack = torch.stack(
                [audio_hidden_0, audio_hidden_1, audio_hidden_2, audio_hidden_3],
                dim=0,
            )
        else:
            # 5 patterns: unconditional, anchor, audio_self, audio_other, all
            anchor_hidden_input = torch.cat(
                [anchor_zeros, anchor_hidden, anchor_zeros, anchor_zeros, anchor_hidden],
                dim=0,
            )
            audio_hidden_stack = torch.stack(
                [audio_hidden_0, audio_hidden_0, audio_hidden_1, audio_hidden_2, audio_hidden_3],
                dim=0,
            )
        face_outputs = []
        use_kv_cache = getattr(self, '_use_kv_cache', True)
        use_compiled = use_kv_cache and getattr(self, '_use_torch_compile_gpt_loop', False)

        # Lazy-compile the GPT loop functions on first use
        if use_compiled and not hasattr(self, '_compiled_gpt_full'):
            self._compiled_gpt_full = torch.compile(
                _gpt_blocks_full, mode='max-autotune-no-cudagraphs')
            self._compiled_gpt_incr = torch.compile(
                _gpt_blocks_incr, mode='max-autotune-no-cudagraphs')

        kv_caches = [None] * len(self.blocks)        # dict-based path
        c_srk = c_srv = c_spk = c_spv = None  # flat cache for compiled path
        for t in range(self.inpainting_length, seq_len):
            is_first = (c_srk is None) if use_compiled else (kv_caches[0] is None)
            if is_first or not use_kv_cache:
                # Full forward (first iteration or cache disabled)
                x = face_hidden[:, :t].repeat(n_cfg, 1, 1)
                audio_hidden_input = audio_hidden_stack[:, :, :t].reshape(n_cfg * bs, t, -1)
                if use_compiled:
                    x, c_srk, c_srv, c_spk, c_spv = self._compiled_gpt_full(
                        self.blocks, x, audio_hidden_input, anchor_hidden_input)
                elif use_kv_cache:
                    for i, block in enumerate(self.blocks):
                        x, kv_caches[i] = _block_forward_full(
                            block, x, audio_hidden_input, anchor_hidden_input)
                else:
                    for block in self.blocks:
                        x = block(
                            x, audio_hidden_input, anchor_hidden_input,
                            causal_mask[:t, :t], cross_causal_mask[:t, :t])
                x_out = x[:, -1:]
            else:
                # Incremental: only process the new token
                x_new = face_hidden[:, t-1:t].repeat(n_cfg, 1, 1)
                audio_new = audio_hidden_stack[:, :, t-1:t].reshape(n_cfg * bs, 1, -1)
                if use_compiled:
                    x_new, c_srk, c_srv, c_spk, c_spv = self._compiled_gpt_incr(
                        self.blocks, x_new, audio_new, anchor_hidden_input,
                        c_srk, c_srv, c_spk, c_spv, t - 1)
                else:
                    for i, block in enumerate(self.blocks):
                        x_new, kv_caches[i] = _block_forward_incr(
                            block, x_new, audio_new, anchor_hidden_input,
                            kv_caches[i], offset=t-1)
                x_out = x_new
            x_t = self.output_norm(x_out)
            gpt_output_t = self.output_proj(x_t)
            if noise_scheduler is not None:
                latent_t = torch.randn_like(gpt_output_t[:bs])
                for i, timestep in enumerate(timesteps):
                    t_batch = torch.full((bs,), timestep, device=device, dtype=torch.long)
                    latent_model_input = latent_t
                    time_embedding = self.time_embed(t_batch).unsqueeze(1)
                    output_batch = self.diffusion_head(
                        latent_model_input,
                        gpt_output_t,
                        temb=time_embedding,
                    )
                    if skip_anchor:
                        noise_pred_uncond, noise_pred_cond_audio, noise_pred_cond_audio_other, noise_pred_cond_all = output_batch.chunk(4, dim=0)
                        noise_pred = (
                            noise_pred_uncond
                            + self.cfg_audio * (noise_pred_cond_audio - noise_pred_uncond)
                            + self.cfg_audio_other * (noise_pred_cond_audio_other - noise_pred_uncond)
                            + self.cfg_all * (noise_pred_cond_all - noise_pred_uncond)
                        )
                    else:
                        noise_pred_uncond, noise_pred_cond_anchor, noise_pred_cond_audio, noise_pred_cond_audio_other, noise_pred_cond_all = output_batch.chunk(5, dim=0)
                        noise_pred = (
                            noise_pred_uncond
                            + self.cfg_audio * (noise_pred_cond_audio - noise_pred_uncond)
                            + self.cfg_audio_other * (noise_pred_cond_audio_other - noise_pred_uncond)
                            + self.cfg_anchor * (noise_pred_cond_anchor - noise_pred_uncond)
                            + self.cfg_all * (noise_pred_cond_all - noise_pred_uncond)
                        )
                    sigma_idx = noise_scheduler.step_index
                    if sigma_idx is None:
                        noise_scheduler._init_step_index(timestep)
                        sigma_idx = noise_scheduler.step_index
                    sigma = noise_scheduler.sigmas[sigma_idx].to(device=device)
                    velocity = (latent_t - noise_pred) / (sigma + 1e-9)
                    latent_t = noise_scheduler.step(
                        velocity, timestep, latent_t, return_dict=False
                    )[0]
                denoised_output_t = latent_t
            face_outputs.append(denoised_output_t)
            if t < seq_len:
                face_hidden[:, t] = self.face_embed(denoised_output_t.squeeze(1))
        output = torch.cat(face_outputs, dim=1)
        return output

    def inference(
        self,
        audio,
        audio_other=None,
        init_motion=None,
        cond_motion=None,
        anchor_motion=None,
        noise_scheduler=None,
        num_inference_steps=10,
    ):
        inpainting_length = self.inpainting_length
        length = cond_motion.shape[1]
        bs = audio.shape[0] if audio is not None else audio_other.shape[0]
        device = audio.device if audio is not None else audio_other.device
        fake_motion = torch.zeros(bs, length, self.cfg.vae_codebook_size).to(device)
        if cond_motion is not None:
            fake_motion[:, : cond_motion.shape[1]] = cond_motion
        cond_motion = fake_motion
        generator = torch.Generator(device=device)
        generator.manual_seed(self.cfg.seed)
        bs, total_len, c = cond_motion.shape
        window = self.cfg.cbh_window_length
        pre_frames = self.inpainting_length
        prev_audio_frames = self.cfg.prev_audio_frames
        stride = 1
        rec_all_face = []
        past_motion = cond_motion[:, :pre_frames, :]
        past_audio = audio[:, : pre_frames * (self.cfg.audio_fps // self.cfg.pose_fps)]
        past_audio_other = audio_other[:, : pre_frames * (self.cfg.audio_fps // self.cfg.pose_fps)]
        past_audio_self = None
        past_audio_other = None
        rec_all_face.append(past_motion[:, :inpainting_length])
        audio_list = [i.cpu().numpy() for i in audio]
        inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio2face_fea = self.audio_encoder_face(
            torch.concat([inputs.input_values, torch.zeros([1, 80], device=inputs.input_values.device)], dim=-1)
        )["high_level"]
        audio2face_fea = F.interpolate(
            audio2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps / 50), mode="linear", align_corners=True
        ).transpose(1, 2)
        audio_other_list = [i.cpu().numpy() for i in audio_other]
        inputs = self.audio_processor(audio_other_list, sampling_rate=16000, return_tensors="pt", padding=True).to(audio.device)
        audio_other2face_fea = self.audio_encoder_face_other(
            torch.concat([inputs.input_values, torch.zeros([1, 80], device=inputs.input_values.device)], dim=-1)
        )["high_level"]
        audio_other2face_fea = F.interpolate(
            audio_other2face_fea.transpose(1, 2), scale_factor=(self.cfg.pose_fps / 50), mode="linear", align_corners=True
        ).transpose(1, 2)
        for i in range(0, total_len, stride):
            start_idx = i
            end_idx = min(start_idx + window, total_len)
            window_size = end_idx - start_idx
            if window_size < window:
                break
            audio_slice_len = window_size * (self.cfg.audio_fps // self.cfg.pose_fps)
            audio_slice_start = start_idx * (self.cfg.audio_fps // self.cfg.pose_fps)
            audio_slice = audio[:, audio_slice_start : audio_slice_start + audio_slice_len] if audio is not None else None
            audio_slice_other = audio_other[:, audio_slice_start : audio_slice_start + audio_slice_len] if audio_other is not None else None
            out = self.one_clip_only_inference(
                per_compute_audio_feature=audio2face_fea[:, start_idx:end_idx],
                per_compute_audio_other_feature=audio_other2face_fea[:, start_idx:end_idx],
                past_audio_self=past_audio,
                audio_self=audio_slice,
                past_audio_other=past_audio_other,
                audio_other=audio_slice_other,
                past_motion=past_motion,
                gen_frames=stride,
                anchor_latent=anchor_motion,
                noise_scheduler=noise_scheduler,
                num_inference_steps=num_inference_steps,
            )
            face_latent = out
            past_motion = torch.concat([past_motion, out], dim=1)[:, -inpainting_length:]
            past_audio = audio_slice[:, : -stride * (self.cfg.audio_fps // self.cfg.pose_fps)]
            rec_all_face.append(face_latent)
        rec_all_face = torch.cat(rec_all_face, dim=1)
        return rec_all_face
