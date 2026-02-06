import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any

from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import Timesteps, TimestepEmbedding, get_1d_rotary_pos_embed
from diffusers.models.normalization import FP32LayerNorm
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class CustomAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CustomAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    @staticmethod
    def _apply_rotary(hidden: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        if hidden.shape[2] < freqs.shape[2]:
            freqs = freqs[:, :, :hidden.shape[2], :]
        
        if hidden.shape[0] != freqs.shape[0]:
            if hidden.shape[0] % freqs.shape[0] != 0:
                raise ValueError("Batch size of hidden states must be a multiple of batch size of freqs.")
            num_repeats = hidden.shape[0] // freqs.shape[0]
            freqs = freqs.repeat_interleave(num_repeats, 0)

        x_complex = torch.view_as_complex(hidden.to(torch.float64).unflatten(3, (-1, 2)))
        x_rot = torch.view_as_real(x_complex * freqs).flatten(3, 4)
        return x_rot.type_as(hidden)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # If no encoder states provided, use self-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Project to QKV
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Optional query/key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Reshape for multi-head: (batch, heads, seq_len, head_dim)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # Apply rotary embeddings if provided
        if rotary_emb is not None:
            query = self._apply_rotary(query, rotary_emb)
            key = self._apply_rotary(key, rotary_emb)
        if pos_emb is not None:
            query = query + pos_emb[:, :, :query.shape[2], :]
            key = key + pos_emb[:, :, :key.shape[2], :]
            value = value + pos_emb[:, :, :value.shape[2], :]

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Merge heads and project out
        hidden_states = attn_output.transpose(1, 2).flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CustomAttnProcessor2_0_distill:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CustomAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    @staticmethod
    def _apply_rotary(hidden: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        if hidden.shape[2] < freqs.shape[2]:
            freqs = freqs[:, :, :hidden.shape[2], :]
        
        if hidden.shape[0] != freqs.shape[0]:
            if hidden.shape[0] % freqs.shape[0] != 0:
                raise ValueError("Batch size of hidden states must be a multiple of batch size of freqs.")
            num_repeats = hidden.shape[0] // freqs.shape[0]
            freqs = freqs.repeat_interleave(num_repeats, 0)

        x_complex = torch.view_as_complex(hidden.to(torch.float64).unflatten(3, (-1, 2)))
        x_rot = torch.view_as_real(x_complex * freqs).flatten(3, 4)
        return x_rot.type_as(hidden)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        distill: bool = False,
    ) -> torch.Tensor:
        # If no encoder states provided, use self-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Project to QKV
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Optional query/key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Reshape for multi-head: (batch, heads, seq_len, head_dim)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # Apply rotary embeddings if provided
        if rotary_emb is not None:
            query = self._apply_rotary(query, rotary_emb)
            key = self._apply_rotary(key, rotary_emb)
        if pos_emb is not None:
            query = query + pos_emb[:, :, :query.shape[2], :]
            key = key + pos_emb[:, :, :key.shape[2], :]
            value = value + pos_emb[:, :, :value.shape[2], :]
        batch_size, heads, query_len, head_dim = query.shape
        if distill:

            scale = 1 / (head_dim ** 0.5)
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
 
            attention_probs = F.softmax(attention_scores, dim=-1)

            hidden_states = torch.matmul(attention_probs, value)
            

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            
            return hidden_states, attention_probs
            
        else:
            # Scaled dot-product attention
            attn_output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            # Merge heads and project out
            hidden_states = attn_output.transpose(1, 2).flatten(2, 3).type_as(query)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

class CustomAttnProcessor2_0_Prev:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CustomAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    @staticmethod
    def _apply_rotary(hidden: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        if hidden.shape[2] < freqs.shape[2]:
            freqs = freqs[:, :, :hidden.shape[2], :]
        
        if hidden.shape[0] != freqs.shape[0]:
            if hidden.shape[0] % freqs.shape[0] != 0:
                raise ValueError("Batch size of hidden states must be a multiple of batch size of freqs.")
            num_repeats = hidden.shape[0] // freqs.shape[0]
            freqs = freqs.repeat_interleave(num_repeats, 0)
            
        x_complex = torch.view_as_complex(hidden.to(torch.float64).unflatten(3, (-1, 2)))
        x_rot = torch.view_as_real(x_complex * freqs).flatten(3, 4)
        return x_rot.type_as(hidden)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # If no encoder states provided, use self-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Project to QKV
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Optional query/key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Reshape for multi-head: (batch, heads, seq_len, head_dim)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # Apply rotary embeddings if provided
        key_len, query_len = key.shape[2], query.shape[2]
        # print(rotary_emb.shape)
        if rotary_emb is not None:
            query = self._apply_rotary(query, rotary_emb[:, :, key_len:, :])
            key = self._apply_rotary(key, rotary_emb[:, :, :key_len, :])

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Merge heads and project out
        hidden_states = attn_output.transpose(1, 2).flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

# reproduce infp
class INFPTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=None,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=None,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=None,
            added_proj_bias=False,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) 

        # 3. Self-attention-2
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn3 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=None,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=None,
            added_proj_bias=False,
            processor=CustomAttnProcessor2_0(),
        )

        # 4. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        # temb: torch.Tensor,
        rotary_emb: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, pos_emb=pos_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, pos_emb=pos_emb, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output

        # 3. Cross-attention-2
        norm_hidden_states = self.norm3(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn3(hidden_states=norm_hidden_states, encoder_hidden_states=torch.cat([global_anchor, prev_motion], dim=1), pos_emb=pos_emb, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output

        # 4. Feed-forward
        norm_hidden_states = self.norm4(hidden_states.float()).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = hidden_states + ff_output
        return hidden_states


class INFP_Step2_TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Cross-attention-2        
        self.attn3 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False) if cross_attn_norm else nn.Identity()

        # 4. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        # temb: torch.Tensor,
        rotary_emb: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, pos_emb=pos_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, pos_emb=pos_emb, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output

        # 3. Cross-attention-2
        norm_hidden_states = self.norm3(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn3(hidden_states=norm_hidden_states, encoder_hidden_states=torch.cat([global_anchor, prev_motion], dim=1), pos_emb=pos_emb, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output

        # 4. Feed-forward
        norm_hidden_states = self.norm4(hidden_states.float()).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = hidden_states + ff_output
        return hidden_states


class INFP_Step1_TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=None,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=None,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=None,
            added_proj_bias=False,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) 

        # 3. Self-attention-2
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn3 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=None,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=None,
            added_proj_bias=False,
            processor=CustomAttnProcessor2_0(),
        )

        # 4. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 5. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(temb).chunk(6, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output

        # 3. Cross-attention-2
        norm_hidden_states = self.norm3(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn3(hidden_states=norm_hidden_states, encoder_hidden_states=torch.cat([global_anchor, prev_motion], dim=1), rotary_emb=rotary_emb, attention_mask=attn3_mask)
        hidden_states = hidden_states + attn_output

        # 4. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)
        return hidden_states


class INFP_DIT_TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Cross-attention-2        
        self.attn3 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False) if cross_attn_norm else nn.Identity()

        # 4. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 5. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(temb).chunk(6, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output

        # 3. Cross-attention-2
        norm_hidden_states = self.norm3(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn3(hidden_states=norm_hidden_states, encoder_hidden_states=torch.cat([global_anchor, prev_motion], dim=1), rotary_emb=rotary_emb, attention_mask=attn3_mask)
        hidden_states = hidden_states + attn_output

        # 4. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)
        return hidden_states


class CustomTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Self-attention-2
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn3 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 4. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 5. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(temb).chunk(6, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=torch.cat([global_anchor, norm_hidden_states], dim=1), rotary_emb=rotary_emb, attention_mask=attn1_mask)[:, global_anchor.size(1):]
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output

        # 3. Self-attention-2
        norm_hidden_states = self.norm3(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn3(hidden_states=torch.cat([prev_motion, norm_hidden_states], dim=1), pos_emb=pos_emb, attention_mask=attn3_mask)[:, prev_motion.size(1):]
        hidden_states = hidden_states + attn_output

        # 4. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states
        

class CustomTransformerBlock_Global(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 1.1 Cross-attention with global anchor
        self.attn1_1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm1_1 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 1.2 Cross-attention with prev motion
        self.attn1_2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0_Prev(),
        )
        # self.norm1_2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 4. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 8 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_attn_1_1, gate_attn_1_2 = self.adaLN_modulation(temb).chunk(8, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 1.1 Cross-attention with global anchor
        norm_hidden_states = self.norm1_1(hidden_states.float()).type_as(hidden_states)
        attn_output_1_1 = self.attn1_1(hidden_states=norm_hidden_states, encoder_hidden_states=global_anchor, rotary_emb=None, pos_emb=None, attention_mask=None) 
        # 1.2 Cross-attention with prev motion
        attn_output_1_2 = self.attn1_2(hidden_states=norm_hidden_states, encoder_hidden_states=prev_motion, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output_1_1 * gate_attn_1_1 + attn_output_1_2 * gate_attn_1_2

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states
        

class CustomTransformerBlock_Global_Gate(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 1.1 Cross-attention with global anchor
        self.attn1_1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm1_1 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 1.2 Cross-attention with prev motion
        self.attn1_2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0_Prev(),
        )
        # self.norm1_2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 4. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_attn_1_1, gate_attn_1_2, gate_attn_audio = self.adaLN_modulation(temb).chunk(9, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 1.1 Cross-attention with global anchor
        norm_hidden_states = self.norm1_1(hidden_states.float()).type_as(hidden_states)
        attn_output_1_1 = self.attn1_1(hidden_states=norm_hidden_states, encoder_hidden_states=global_anchor, rotary_emb=None, pos_emb=None, attention_mask=None) 
        # 1.2 Cross-attention with prev motion
        attn_output_1_2 = self.attn1_2(hidden_states=norm_hidden_states, encoder_hidden_states=prev_motion, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output_1_1 * gate_attn_1_1 + attn_output_1_2 * gate_attn_1_2

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output * gate_attn_audio

        # 3. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states



class CustomTransformerBlock_GlobalOrder(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 1.1 Cross-attention with global anchor
        self.attn1_1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm1_1 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 1.2 Cross-attention with prev motion
        self.attn1_2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0_Prev(),
        )
        # self.norm1_2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 4. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 8 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_attn_1_1, gate_attn_1_2 = self.adaLN_modulation(temb).chunk(8, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
        
        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output

        # 1.1 Cross-attention with global anchor
        norm_hidden_states = self.norm1_1(hidden_states.float()).type_as(hidden_states)
        attn_output_1_1 = self.attn1_1(hidden_states=norm_hidden_states, encoder_hidden_states=global_anchor, rotary_emb=None, pos_emb=None, attention_mask=None) 
        # 1.2 Cross-attention with prev motion
        attn_output_1_2 = self.attn1_2(hidden_states=norm_hidden_states, encoder_hidden_states=prev_motion, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output_1_1 * gate_attn_1_1 + attn_output_1_2 * gate_attn_1_2

        # 3. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states


class CustomTransformerBlock_SA(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 4. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(temb).chunk(6, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        norm_hidden_states = torch.cat([global_anchor, prev_motion, norm_hidden_states], dim=1)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)[:, -hidden_states.shape[1]:]
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
        
        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output 

        # 3. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states


class CustomTransformerBlock_GlobalOrder_NoGate(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 1.1 Cross-attention with global anchor
        self.attn1_1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm1_1 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 1.2 Cross-attention with prev motion
        self.attn1_2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0_Prev(),
        )
        # self.norm1_2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 4. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(temb).chunk(6, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
        
        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output

        # 1.1 Cross-attention with global anchor
        norm_hidden_states = self.norm1_1(hidden_states.float()).type_as(hidden_states)
        attn_output_1_1 = self.attn1_1(hidden_states=norm_hidden_states, encoder_hidden_states=global_anchor, rotary_emb=None, pos_emb=None, attention_mask=None) 
        # 1.2 Cross-attention with prev motion
        attn_output_1_2 = self.attn1_2(hidden_states=norm_hidden_states, encoder_hidden_states=prev_motion, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output_1_1 + attn_output_1_2

        # 3. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states




class CustomTransformerBlock_Deta(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 1.1 Cross-attention with global anchor
        self.attn1_1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm1_1 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 1.2 Cross-attention with prev motion
        self.attn1_2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0_Prev(),
        )
        # self.norm1_2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 4. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 8 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_attn_1_1, gate_attn_1_2 = self.adaLN_modulation(temb).chunk(8, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
        
        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output

        # 1.1 Cross-attention with global anchor
        norm_hidden_states = self.norm1_1(hidden_states.float()).type_as(hidden_states)
        attn_output_1_1 = self.attn1_1(hidden_states=norm_hidden_states, encoder_hidden_states=global_anchor, rotary_emb=rotary_emb, attention_mask=None) 
        # 1.2 Cross-attention with prev motion
        attn_output_1_2 = self.attn1_2(hidden_states=norm_hidden_states, encoder_hidden_states=prev_motion, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output_1_1 * gate_attn_1_1 + attn_output_1_2 * gate_attn_1_2

        # 3. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states



class CustomTransformerBlock_GlobalOrder_Gate(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 1.1 Cross-attention with global anchor
        self.attn1_1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm1_1 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 1.2 Cross-attention with prev motion
        self.attn1_2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0_Prev(),
        )
        # self.norm1_2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 4. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_attn_1_1, gate_attn_1_2, gate_attn_audio = self.adaLN_modulation(temb).chunk(9, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
        
        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output * gate_attn_audio

        # 1.1 Cross-attention with global anchor
        norm_hidden_states = self.norm1_1(hidden_states.float()).type_as(hidden_states)
        attn_output_1_1 = self.attn1_1(hidden_states=norm_hidden_states, encoder_hidden_states=global_anchor, rotary_emb=None, pos_emb=None, attention_mask=None) 
        # 1.2 Cross-attention with prev motion
        attn_output_1_2 = self.attn1_2(hidden_states=norm_hidden_states, encoder_hidden_states=prev_motion, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output_1_1 * gate_attn_1_1 + attn_output_1_2 * gate_attn_1_2

        # 3. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states


class CustomTransformerBlock_Allrope(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 1.1 Cross-attention with global anchor
        self.attn1_1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm1_1 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 1.2 Cross-attention with prev motion
        self.attn1_2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        # self.norm1_2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 4. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 8 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_attn_1_1, gate_attn_1_2 = self.adaLN_modulation(temb).chunk(8, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 1.1 Cross-attention with global anchor
        norm_hidden_states = self.norm1_1(hidden_states.float()).type_as(hidden_states)
        attn_output_1_1 = self.attn1_1(hidden_states=norm_hidden_states, encoder_hidden_states=global_anchor, rotary_emb=rotary_emb, attention_mask=None) 
        # 1.2 Cross-attention with prev motion
        attn_output_1_2 = self.attn1_2(hidden_states=norm_hidden_states, encoder_hidden_states=prev_motion, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output_1_1 * gate_attn_1_1 + attn_output_1_2 * gate_attn_1_2

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states

class CustomTransformerBlock_Allrope_order(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 1.1 Cross-attention with global anchor
        self.attn1_1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm1_1 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 1.2 Cross-attention with prev motion
        self.attn1_2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        # self.norm1_2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 4. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 8 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_attn_1_1, gate_attn_1_2 = self.adaLN_modulation(temb).chunk(8, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output

        # 1.1 Cross-attention with global anchor
        norm_hidden_states = self.norm1_1(hidden_states.float()).type_as(hidden_states)
        attn_output_1_1 = self.attn1_1(hidden_states=norm_hidden_states, encoder_hidden_states=global_anchor, rotary_emb=rotary_emb, attention_mask=None) 
        # 1.2 Cross-attention with prev motion
        attn_output_1_2 = self.attn1_2(hidden_states=norm_hidden_states, encoder_hidden_states=prev_motion, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output_1_1 * gate_attn_1_1 + attn_output_1_2 * gate_attn_1_2

        # 3. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states


class CustomTransformerBlock_Allrope_order_nogate(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 1.1 Cross-attention with global anchor
        self.attn1_1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm1_1 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 1.2 Cross-attention with prev motion
        self.attn1_2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        # self.norm1_2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 4. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(temb).chunk(6, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output

        # 1.1 Cross-attention with global anchor
        norm_hidden_states = self.norm1_1(hidden_states.float()).type_as(hidden_states)
        attn_output_1_1 = self.attn1_1(hidden_states=norm_hidden_states, encoder_hidden_states=global_anchor, rotary_emb=rotary_emb, attention_mask=None) 
        # 1.2 Cross-attention with prev motion
        attn_output_1_2 = self.attn1_2(hidden_states=norm_hidden_states, encoder_hidden_states=prev_motion, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output_1_1 + attn_output_1_2

        # 3. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states


class CustomTransformerBlock_Allrope_order_nogate_seq(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 1.1 Cross-attention with global anchor
        self.attn1_1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm1_1 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 1.2 Cross-attention with prev motion
        self.attn1_2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm1_2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 4. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(temb).chunk(6, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output

        # 1.1 Cross-attention with global anchor
        norm_hidden_states = self.norm1_1(hidden_states.float()).type_as(hidden_states)
        attn_output_1_1 = self.attn1_1(hidden_states=norm_hidden_states, encoder_hidden_states=global_anchor, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output_1_1
        # 1.2 Cross-attention with prev motion
        norm_hidden_states = self.norm1_2(hidden_states.float()).type_as(hidden_states)
        attn_output_1_2 = self.attn1_2(hidden_states=norm_hidden_states, encoder_hidden_states=prev_motion, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output_1_2

        # 3. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states


class CustomTransformerBlock_Halfrope(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CustomAttnProcessor2_0(),
        )

        # 1.1 Cross-attention with global anchor
        self.attn1_1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm1_1 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 1.2 Cross-attention with prev motion
        self.attn1_2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        # self.norm1_2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=CustomAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm4 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 4. time embedding adain
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 8 * dim, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None,
        global_anchor: Optional[torch.Tensor] = None,
        attn1_mask: Optional[torch.Tensor] = None,
        attn2_mask: Optional[torch.Tensor] = None,
        attn3_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_attn_1_1, gate_attn_1_2 = self.adaLN_modulation(temb).chunk(8, dim=-1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, attention_mask=attn1_mask)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 1.1 Cross-attention with global anchor
        norm_hidden_states = self.norm1_1(hidden_states.float()).type_as(hidden_states)
        attn_output_1_1 = self.attn1_1(hidden_states=norm_hidden_states, encoder_hidden_states=global_anchor, rotary_emb=None, attention_mask=None) 
        # 1.2 Cross-attention with prev motion
        attn_output_1_2 = self.attn1_2(hidden_states=norm_hidden_states, encoder_hidden_states=prev_motion, rotary_emb=rotary_emb, attention_mask=None)
        hidden_states = hidden_states + attn_output_1_1 * gate_attn_1_1 + attn_output_1_2 * gate_attn_1_2

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states, rotary_emb=rotary_emb, attention_mask=attn2_mask)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm4(hidden_states.float()) * (1 + scale_mlp) + shift_mlp).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * gate_mlp).type_as(hidden_states)

        return hidden_states


class CustomConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, seed_frames=None, **kwargs):
        self.pad = (kernel_size - 1) // 2 * dilation
        self.seed_frames = seed_frames
        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            **kwargs
        )

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        return super().forward(x)[:, :, self.seed_frames:]


def zero_adaptive_instance_normalization(content, style, eps=1e-5):
    assert len(content.shape) == 3
    mean_content = content.mean(dim=1, keepdim=True)
    std_content = content.std(dim=1, keepdim=True)
    mean_style = style.mean(dim=1, keepdim=True)
    std_style = style.std(dim=1, keepdim=True)
    normalized_content = (content - mean_content) / (std_content+eps)
    return normalized_content * (1+std_style) + mean_style

class WanTimeEmbedding(nn.Module):
    """
    Modified from:
    Wan: Open and Advanced Large-Scale Video Generative Models
    https://huggingface.co/docs/diffusers/main/api/models/wan_transformer_3d
    """
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
    ):
        super().__init__()
        # generate sinusoidal time embeddings
        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        # project to model dimension
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim,
            time_embed_dim=dim,
        )

    def forward(self, timestep: torch.Tensor):  # timestep: (batch,)
        # 1. sinusoidal embedding: (batch, time_freq_dim)
        timestep = self.timesteps_proj(timestep)
        # ensure dtype matches embedder
        emb_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != emb_dtype and emb_dtype != torch.int8:
            timestep = timestep.to(emb_dtype)
        # 2. linear + activation: (batch, dim)
        temb = self.time_embedder(timestep)
        return temb


class CustomPosEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 128,
    ):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1), :].unsqueeze(1)


class WanRotaryPosEmbed(nn.Module):
    """
    Modified from:
    Wan: Open and Advanced Large-Scale Video Generative Models
    https://huggingface.co/docs/diffusers/main/api/models/wan_transformer_3d
    """
    def __init__(
        self,
        attention_head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.max_seq_len = max_seq_len
        # 1D rotary embedding: only time dimension
        self.freqs = get_1d_rotary_pos_embed(
            attention_head_dim,
            max_seq_len,
            theta,
            use_real=False,
            repeat_interleave_real=False,
            freqs_dtype=torch.float64,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, num_frames, num_channels)
        batch_size, num_frames, num_channels = hidden_states.shape
        # 1D rotary embedding: expand to (batch, heads, num_frames, head_dim)
        freqs = self.freqs.to(hidden_states.device)
        freqs = freqs[:num_frames]  # (num_frames, head_dim)
        # Expand to match attention shapes
        freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, num_frames, head_dim)
        return freqs


class LIAMemoryBank(nn.Module):
    """
    Modified from:
    LIA: Latent Image Animator
    https://github.com/wyhsirius/LIA
    """
    def __init__(self, latent_dim, num_direction):
        super(LIAMemoryBank, self).__init__()
        self.weight = nn.Parameter(torch.randn(latent_dim, num_direction))

    def forward(self, input):
        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)
            return out


class TANGOWrapedWav2Vec(nn.Module):
    """
    Modified from:
    TANGO: Co-Speech Gesture Video Reenactment with Hierarchical Audio-Motion Embedding and Diffusion Interpolation
    https://github.com/CyberAgentAILab/TANGO
    """
    def __init__(self, layers: int = 1):
        super(TANGOWrapedWav2Vec, self).__init__()
        base = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.feature_extractor = base.feature_extractor
        self.feature_projection = base.feature_projection
        self.encoder = base.encoder
        self.encoder.layers = self.encoder.layers[:layers]

    def forward(
        self,
        inputs,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        finetune_audio_low = self.feature_extractor(inputs).transpose(1, 2)
        hidden_states, _ = self.feature_projection(finetune_audio_low.detach())
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        return {"low_level": finetune_audio_low, "high_level": hidden_states}


class TANGOWrapedWav2Vec_Double(nn.Module):
    """
    Modified from:
    TANGO: Co-Speech Gesture Video Reenactment with Hierarchical Audio-Motion Embedding and Diffusion Interpolation
    https://github.com/CyberAgentAILab/TANGO
    """
    def __init__(self, layers: int = 1):
        super(TANGOWrapedWav2Vec_Double, self).__init__()
        base = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.feature_extractor = base.feature_extractor
        self.feature_projection = base.feature_projection
        self.encoder = base.encoder
        self.encoder.layers = self.encoder.layers[:layers]

        base_freeze = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.feature_extractor_freeze = base_freeze.feature_extractor
        self.feature_projection_freeze = base_freeze.feature_projection
        self.encoder_freeze = base_freeze.encoder
        self.encoder_freeze.layers = self.encoder_freeze.layers[:layers]

    def forward(
        self,
        inputs,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        finetune_audio_low = self.feature_extractor(inputs).transpose(1, 2)
        hidden_states, _ = self.feature_projection(finetune_audio_low.detach())
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        finetune_audio_low_freeze = self.feature_extractor_freeze(inputs).transpose(1, 2)
        hidden_states_freeze, _ = self.feature_projection_freeze(finetune_audio_low_freeze)
        encoder_outputs_freeze = self.encoder_freeze(
            hidden_states_freeze,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states_freeze = encoder_outputs_freeze[0]
        hierarchical_states = torch.cat([finetune_audio_low, finetune_audio_low_freeze, hidden_states, hidden_states_freeze], dim=-1)
        return {"low_level": finetune_audio_low, "high_level": hierarchical_states}


class FreezeWrapedWav2Vec(nn.Module):
    """
    Modified from:
    TANGO: Co-Speech Gesture Video Reenactment with Hierarchical Audio-Motion Embedding and Diffusion Interpolation
    https://github.com/CyberAgentAILab/TANGO
    """
    def __init__(self, layers: int = 1):
        super(FreezeWrapedWav2Vec, self).__init__()
        base = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.feature_extractor = base.feature_extractor
        self.feature_projection = base.feature_projection
        self.encoder = base.encoder
        self.encoder.layers = self.encoder.layers[:layers]

    def forward(
        self,
        inputs,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        finetune_audio_low = self.feature_extractor(inputs).transpose(1, 2)
        hidden_states, _ = self.feature_projection(finetune_audio_low)
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0].detach()
        return {"low_level": finetune_audio_low, "high_level": hidden_states}


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(channel, channel, 3, 1, 1),
        )
    def forward(self, x):
        return self.model(x)+x

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
class TM2TVQEncoderV5(nn.Module):
    """
    Modified from:
    TM2T: Stochastical and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts
    https://github.com/EricGuo5513/TM2T
    """
    def __init__(self, input_size=512, n_down=2, channels=[768, 768, 768]):
        super().__init__()
        self.input_size = input_size
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            ResBlock(channels[0]),
        ]
        for i in range(1, n_down+1):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        outputs = self.main(inputs).permute(0,2,1)
        return outputs

