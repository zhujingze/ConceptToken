import types
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaAttention, repeat_kv
import types
from typing import Optional, Tuple, TypedDict
from transformers.cache_utils import Cache
import sys
import typing_extensions
if sys.version_info >= (3, 11):
    Unpack = typing.Unpack
else:
    Unpack = typing_extensions.Unpack
import math
def llama_new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)


    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    
    if hasattr(self, "use_attn"):
        use_attn = self.use_attn
        first_token_idx = self.first_token_idx
        token_enhance = self.token_enhance
        token_weaken = self.token_weaken
    else:
        use_attn = False
    before = attn_weights.clone()
    #print('b',attn_weights[:, :, -1:, token_weaken[0]])
    #attn_weights_b = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    #print('b',attn_weights_b[:, :, first_token_idx:, token_weaken[0]])
    if use_attn:
        if token_enhance:
            attn_weights[:, :, first_token_idx:, token_enhance] = (
                attn_weights[:, :, first_token_idx:, token_enhance].abs() * self.alpha
                + attn_weights[:, :, first_token_idx:, token_enhance]
            )
        if token_weaken:
            #print('do weaken')
            attn_weights[:, :, first_token_idx:, token_weaken] = (
                - attn_weights[:, :, first_token_idx:, token_weaken].abs() * self.alpha
                + attn_weights[:, :, first_token_idx:, token_weaken]
            )
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    #print('a',attn_weights[:, :, first_token_idx:, token_weaken[0]])
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def llama_modify(
    model,
    model_name: str,
    start_layer: int,
    end_layer: int,
    use_attn: bool,
    alpha: float,
    first_token_idx: int,
    token_enhance: list,
    token_weaken: list,
):
    for i in range(start_layer, end_layer + 1):  # 包含 end_layer
        layer = model.model.layers[i]
        self_attn = layer.self_attn

        # 注入自定义属性
        self_attn.use_attn = use_attn
        self_attn.alpha = alpha
        self_attn.first_token_idx = first_token_idx
        self_attn.token_enhance = token_enhance
        self_attn.token_weaken = token_weaken
        model_type = model_name.lower()
        if 'llama' in model_type:
        # 替换 forward 方法
            self_attn.forward = types.MethodType(llama_new_forward, self_attn)
      
        
def llama_restore(model,model_name, start_layer: int, end_layer: int):
    for i in range(start_layer, end_layer + 1):
        layer = model.model.layers[i]
        self_attn = layer.self_attn
        model_type = model_name.lower()
        if 'llama' in model_type:
            self_attn.forward = types.MethodType(LlamaAttention.forward, self_attn)
    
        for attr in ["use_attn", "alpha", "first_token_idx", "token_enhance", "token_weaken"]:
            if hasattr(self_attn, attr):
                delattr(self_attn, attr)