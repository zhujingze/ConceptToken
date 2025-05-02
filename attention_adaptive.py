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
        sink = self.sink
        th = self.th
        ave_token = self.ave_token
        n_idx = self.n_idx
        c_idx = self.c_idx
        s_idx = self.s_idx
        layer_idx = self.layer_idx
        model = self.model
        ema = self.ema
        start_layer = self.start_layer
    else:
        use_attn = False
    attn_weights_sink = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    c_sum = attn_weights_sink[:, :, first_token_idx:, c_idx].sum(dim=-1).mean(dim=1)
    s_sum = attn_weights_sink[:, :, first_token_idx:, s_idx].sum(dim=-1).mean(dim=1)
    n_sum = attn_weights_sink[:, :, first_token_idx:, n_idx].sum(dim=-1).mean(dim=1)
    # ### 每个head
    # pos_mask = c_sum_per_pos > th  # [batch, heads, q_len]
    # # 3. 扩展掩码到token_weaken的key维度
    # final_mask = pos_mask.unsqueeze(-1).expand(-1, -1, -1, len(token_weaken))  # [batch, heads, q_len, n_weaken]
    # # 4. 精确应用掩码到对应位置
    # attn_slice = attn_weights[:, :, first_token_idx:, token_weaken]  # 获取目标切片
    # attn_slice = torch.where(final_mask, -1e2 * torch.ones_like(attn_slice), attn_slice)
    # attn_weights[:, :, first_token_idx:, token_weaken] = attn_slice
    
    #is_above_th = (sum_enhance > th)
    if use_attn:
        if token_enhance:
            #print(attn_weights[:, :, first_token_idx:, token_enhance[0]])
            attn_weights[:, :, first_token_idx:, token_enhance] = (
                attn_weights[:, :, first_token_idx:, token_enhance].abs() * self.alpha
                + attn_weights[:, :, first_token_idx:, token_enhance]
            )
            
        if token_weaken:
            if ema:
                if layer_idx != start_layer:
                    prev_layer = model.model.layers[layer_idx - 1]
                    prev_self_attn = prev_layer.self_attn
                    if hasattr(prev_self_attn, "ave_token"):
                        prev_ave_token = prev_self_attn.ave_token
                        ave_token = c_sum
                        # ave_token[:,1] = p_sum
                        # ave_token[:,2] = n_sum
                        #print('now',ave_token,'pre',prev_ave_token)
                        self.ave_token = (1-0.9)*ave_token.to(model.device) + 0.9*prev_ave_token.to(model.device)
                        #print('eam',self.ave_token)
                    else:
                        raise ValueError(f"Layer {layer_idx} has no prev_ave_token")
                else:
                    self.ave_token = c_sum
                    # ave_token[:,1] = p_sum
                    # ave_token[:,2] = n_sum
                    
                # if ave_token[0] >= th:
                #     attn_weights[:, :, first_token_idx:, token_weaken] = -1e2
                pos_mask = self.ave_token > th  # [batch, q_len]
                # 3. 扩展掩码到所有heads和token_weaken的key维度
                final_mask = pos_mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, q_len, 1]
                final_mask = final_mask.expand(-1, attn_weights.size(1), -1, len(token_weaken))  # [batch, heads, q_len, n_weaken]

                # 4. 应用掩码到所有heads的对应位置
                attn_slice = attn_weights[:, :, first_token_idx:, token_weaken]  # 获取目标切片
                final_mask = final_mask.to(attn_slice.device)
                attn_slice = torch.where(final_mask, -1e2 * torch.ones_like(attn_slice), attn_slice)
                attn_weights[:, :, first_token_idx:, token_weaken] = attn_slice
                
            else:
                #print('Layer:',layer_idx,'C ratio',c_sum)
                #if c_sum >= th:
                pos_mask = c_sum > th  # [batch, q_len]
                # 3. 扩展掩码到所有heads和token_weaken的key维度
                final_mask = pos_mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, q_len, 1]
                final_mask = final_mask.expand(-1, attn_weights.size(1), -1, len(token_weaken))  # [batch, heads, q_len, n_weaken]

                # 4. 应用掩码到所有heads的对应位置
                attn_slice = attn_weights[:, :, first_token_idx:, token_weaken]  # 获取目标切片
                attn_slice = torch.where(final_mask, -1e2 * torch.ones_like(attn_slice), attn_slice)
                attn_weights[:, :, first_token_idx:, token_weaken] = attn_slice
                #attn_weights[:, :, first_token_idx:, token_weaken] = -1e2

            
            #print(attn_weights[:, :, first_token_idx:, token_weaken[0]])
            # attn_weights[:, :, first_token_idx:, token_weaken] = (
            #     - attn_weights[:, :, first_token_idx:, token_weaken].abs() * self.alpha
            #     + attn_weights[:, :, first_token_idx:, token_weaken]
            # )
            # attn_weights[:, :, first_token_idx:, token_weaken] = -1e2
            # 提取目标区域的子张量
            # target_attn = attn_weights[:, :, first_token_idx:, :]
            
            # # 计算每一行的最小值（沿最后一个维度）
            # min_values = target_attn.min(dim=-1, keepdim=True).values
            
            # # 将最小值赋给 token_weaken 位置
            # attn_weights[:, :, first_token_idx:, token_weaken] = min_values.expand_as(
            #     attn_weights[:, :, first_token_idx:, token_weaken]
            # )

        if sink:
            # 提取目标区域（first_token_idx之后的token）
            sink_attn = attn_weights_sink[:, :, first_token_idx:, :]  # 形状: [batch, heads, rows, cols]
            
            # 对 head 维度求平均（合并多头）
            avg_attn = sink_attn.mean(dim=1)  # 形状: [batch, rows, cols]
            #print('attn',avg_attn.size())
            # 统计哪些位置的平均值超过0.3
            over_threshold = avg_attn > 0.2  # 形状: [batch, rows, cols]
            
            # 获取所有满足条件的唯一 token 索引
            if over_threshold.any():
                # 提取所有超过阈值的列索引（cols）并去重
                nonzero_indices = over_threshold.nonzero()  # 返回非零元素的坐标 [num_true, 3]
                col_indices = nonzero_indices[:, 2]         # 提取第三列（cols维度）
                unique_indices = col_indices.unique().tolist()  # 去重并转为列表
            else:
                unique_indices = []
                
            attn_weights[:, :, first_token_idx:, unique_indices] = -1e2
            # if len(unique_indices) != 0:
            #     attn_weights[:, :, first_token_idx:, unique_indices] = (
            #         - attn_weights[:, :, first_token_idx:, unique_indices].abs() * 0.95
            #         + attn_weights[:, :, first_token_idx:, unique_indices]
            #     )
            #print(f"Unique token indices with attention > 0.3: {unique_indices}")

    
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

from typing import List, Optional

def llama_modify_adaptive(
    model,
    model_name: str,
    start_layer: int,
    end_layer: int,
    use_attn: bool,
    sink: bool,
    alpha: float,
    first_token_idx: int,
    token_enhance: list,
    token_weaken: list,
    n_idx:list,
    c_idx:list,
    s_idx:list,
    th: float,
    ema: bool,
    ave_token: Optional[torch.Tensor] = None, ###C,P,N
    special_layers: Optional[List[int]] = None
):
    model_type = model_name.lower()
    if 'llama' not in model_type:
        raise ValueError("此方法仅支持 LLaMA 模型")
    total_layers = len(model.model.layers)
    
    # 主层范围校验
    if not (0 <= start_layer <= end_layer < total_layers):
        raise ValueError(f"无效主层范围 {start_layer}-{end_layer}，模型总层数 {total_layers}")

    # 特殊层校验
    if special_layers:
        invalid_layers = [x for x in special_layers if not 0 <= x < total_layers]
        if invalid_layers:
            raise ValueError(f"非法层号 {invalid_layers}，有效范围 0-{total_layers-1}")

    # ----------------------------
    # 第一部分：处理主循环层
    # ----------------------------
    for i in range(start_layer, end_layer + 1):
        layer = model.model.layers[i]
        self_attn = layer.self_attn
        
        # 通用配置
        self_attn.layer_idx = i  # 当前层索引
        self_attn.model = model  # 模型引用
        self_attn.use_attn = use_attn
        self_attn.model_name = model_name
        self_attn.alpha = alpha
        self_attn.first_token_idx = first_token_idx
        self_attn.token_enhance = token_enhance
        self_attn.token_weaken = token_weaken
        self_attn.ema = ema
        self_attn.th = th
        self_attn.n_idx = n_idx
        self_attn.c_idx = c_idx
        self_attn.s_idx = s_idx
        self_attn.ave_token = ave_token
        self_attn.start_layer = start_layer
        # 如果在 special_layers 中则覆盖 sink 值
        self_attn.sink = (i in special_layers) if special_layers else False

        # 修改前向传播
        self_attn.forward = types.MethodType(llama_new_forward, self_attn)

    # ----------------------------
    # 第二部分：独立处理 special_layers
    # ----------------------------
    if special_layers:
        # 去重处理
        unique_special = list(set(special_layers))
        for i in unique_special:
            # 跳过已处理的主循环层
            if start_layer <= i <= end_layer:
                continue
                
            layer = model.model.layers[i]
            self_attn = layer.self_attn
            self_attn.alpha = alpha
            self_attn.use_attn = True
            self_attn.first_token_idx = first_token_idx
            self_attn.sink = True
            self_attn.token_enhance = None
            self_attn.token_weaken = None
            self_attn.forward = types.MethodType(llama_new_forward, self_attn)
            
            # 确保已注入必要属性（防御性编程）
            # if not hasattr(self_attn, 'use_attn'):
              # 默认值
            # 其他属性按需初始化
      
        
def llama_restore_adaptive(model,model_name, start_layer: int, end_layer: int):
    for i in range(start_layer, end_layer + 1):
        layer = model.model.layers[i]
        self_attn = layer.self_attn
        model_type = model_name.lower()
        if 'llama' in model_type:
            self_attn.forward = types.MethodType(LlamaAttention.forward, self_attn)
    
        for attr in ["use_attn", "alpha", "first_token_idx", "token_enhance", "token_weaken", "sink"]:
            if hasattr(self_attn, attr):
                delattr(self_attn, attr)
