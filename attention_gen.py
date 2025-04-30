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
    else:
        use_attn = False
        
    if hasattr(self, "use_cfg"):
        use_cfg = self.use_cfg
    else:
        use_cfg = False

    #print('b',attn_weights[:, :, -1:, token_weaken[0]])
    attn_weights_sink = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    #print('b',attn_weights_b[:, :, first_token_idx:, token_weaken[0]])
    if use_attn and not use_cfg:
        # print(token_enhance,token_weaken,sink)
        if token_enhance:
            attn_weights[:, :, -1, token_enhance] = (
                attn_weights[:, :, -1, token_enhance].abs() * self.alpha
                + attn_weights[:, :, -1, token_enhance]
            )
        if token_weaken:
            #print(attn_weights[:, :, first_token_idx:, token_weaken[0]])
            # attn_weights[:, :, first_token_idx:, token_weaken] = (
            #     - attn_weights[:, :, first_token_idx:, token_weaken].abs() * self.alpha
            #     + attn_weights[:, :, first_token_idx:, token_weaken]
            # )
            attn_weights[:, :, -1, token_weaken] = -1e2
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
            sink_attn = attn_weights_sink[:, :, -1, :]  # 形状: [batch, heads, rows, cols]
            
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
                
            attn_weights[:, :, -1, unique_indices] = -1e2
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

# def llama_modify(
#     model,
#     model_name: str,
#     start_layer: int,
#     end_layer: int,
#     use_attn: bool,
#     sink: bool,
#     alpha: float,
#     first_token_idx: int,
#     token_enhance: list,
#     token_weaken: list,
# ):
#     for i in range(start_layer, end_layer + 1):  # 包含 end_layer
#         layer = model.model.layers[i]
#         self_attn = layer.self_attn

#         # 注入自定义属性
#         self_attn.use_attn = use_attn
#         self_attn.sink = False
#         self_attn.alpha = alpha
#         self_attn.first_token_idx = first_token_idx
#         self_attn.token_enhance = token_enhance
#         self_attn.token_weaken = token_weaken
#         model_type = model_name.lower()
#         if 'llama' in model_type:
#         # 替换 forward 方法
#             self_attn.forward = types.MethodType(llama_new_forward, self_attn)
            
#     for i in range(2, 3 + 1):  # 包含 end_layer
#         layer = model.model.layers[i]
#         self_attn = layer.self_attn
#         self_attn.use_attn = use_attn
#         self_attn.sink = sink
#         # self_attn.alpha = alpha
#         self_attn.first_token_idx = first_token_idx
#         # self_attn.token_enhance = None
#         # self_attn.token_weaken = None
#         model_type = model_name.lower()
#         if 'llama' in model_type:
#         # 替换 forward 方法
#             self_attn.forward = types.MethodType(llama_new_forward, self_attn)

from typing import List, Optional

def llama_modify(
    model,
    model_name: str,
    start_layer: int,
    end_layer: int,
    use_attn: bool,
    use_cfg: bool,
    sink: bool,
    alpha: float,
    first_token_idx: int,
    token_enhance: list,
    token_weaken: list,
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
        self_attn.use_attn = use_attn
        self_attn.alpha = alpha
        self_attn.first_token_idx = first_token_idx
        self_attn.token_enhance = token_enhance
        self_attn.token_weaken = token_weaken
        self_attn.use_cfg = use_cfg
        
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
            self_attn.use_cfg = use_cfg
            self_attn.forward = types.MethodType(llama_new_forward, self_attn)
            # 确保已注入必要属性（防御性编程）
            # if not hasattr(self_attn, 'use_attn'):
              # 默认值
            # 其他属性按需初始化
      
        
def llama_restore(model,model_name, start_layer: int, end_layer: int):
    for i in range(start_layer, end_layer + 1):
        layer = model.model.layers[i]
        self_attn = layer.self_attn
        model_type = model_name.lower()
        if 'llama' in model_type:
            self_attn.forward = types.MethodType(LlamaAttention.forward, self_attn)
    
        for attr in ["use_attn", "alpha", "first_token_idx", "token_enhance", "token_weaken", "sink"]:
            if hasattr(self_attn, attr):
                delattr(self_attn, attr)
                            
    
import os
from collections import namedtuple

import torch
import yaml
from CFG import CFGLogits

def load_model_args_from_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    ModelArgs = namedtuple("ModelArgs", data["ModelArgs"].keys())
    TrainingArgs = namedtuple("TrainingArgs", data["TrainingArgs"].keys())

    model_args = ModelArgs(**data["ModelArgs"])
    training_args = TrainingArgs(**data["TrainingArgs"])

    return model_args, training_args


def load_llava_model(model_path):
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    return tokenizer, model, image_processor, model


def load_minigpt4_model(cfg_path):
    cfg = MiniGPT4Config(cfg_path)
    model, vis_processor = init_model(cfg)
    # TODO:
    # model.eval()
    return model.llama_tokenizer, model, vis_processor, model.llama_model


def load_shikra_model(yaml_path):
    model_args, training_args = load_model_args_from_yaml(yaml_path)
    model, preprocessor = load_pretrained(model_args, training_args)

    return (
        preprocessor["text"],
        model.to("cuda"),
        preprocessor["image"],
        model.to("cuda"),
    )


class MiniGPT4Config:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.options = None


def load_model(model):
    if model == "llava-1.5":
        model_path = os.path.expanduser("/path/to/llava-v1.5-7b")
        return load_llava_model(model_path)

    elif model == "minigpt4":
        cfg_path = "./minigpt4/eval_config/minigpt4_eval.yaml"
        return load_minigpt4_model(cfg_path)

    elif model == "shikra":
        yaml_path = "./mllm/config/config.yml" 
        return load_shikra_model(yaml_path)

    else:
        raise ValueError(f"Unknown model: {model}")


def prepare_llava_inputs(template, query, image, tokenizer):
    image_tensor = image["pixel_values"][0]
    qu = [template.replace("<question>", q) for q in query]
    batch_size = len(query)

    chunks = [q.split("<ImageHere>") for q in qu]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]

    token_before = (
        tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    token_after = (
        tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * tokenizer.bos_token_id
    )

    img_start_idx = len(token_before[0]) + 1
    img_end_idx = img_start_idx + IMAGE_TOKEN_LENGTH
    image_token = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * IMAGE_TOKEN_INDEX
    )

    input_ids = torch.cat([bos, token_before, image_token, token_after], dim=1)
    kwargs = {}
    kwargs["images"] = image_tensor.half()
    kwargs["input_ids"] = input_ids

    return qu, img_start_idx, img_end_idx, kwargs


def prepare_minigpt4_inputs(template, query, image, model):
    image_tensor = image.to("cuda")
    qu = [template.replace("<question>", q) for q in query]
    batch_size = len(query)

    img_embeds, atts_img = model.encode_img(image_tensor.to("cuda"))
    inputs_embeds, attention_mask = model.prompt_wrap(
        img_embeds=img_embeds, atts_img=atts_img, prompts=qu
    )

    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device=inputs_embeds.device)
        * model.llama_tokenizer.bos_token_id
    )
    bos_embeds = model.embed_tokens(bos)
    atts_bos = attention_mask[:, :1]

    # add 1 for bos token
    img_start_idx = (
        model.llama_tokenizer(
            qu[0].split("<ImageHere>")[0], return_tensors="pt", add_special_tokens=False
        ).input_ids.shape[-1]
        + 1
    )
    img_end_idx = img_start_idx + MINIGPT4_IMAGE_TOKEN_LENGTH

    inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
    attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

    kwargs = {}
    kwargs["inputs_embeds"] = inputs_embeds
    kwargs["attention_mask"] = attention_mask

    return qu, img_start_idx, img_end_idx, kwargs


def prepare_shikra_inputs(template, query, image, tokenizer):
    image_tensor = image["pixel_values"][0]

    replace_token = DEFAULT_IMAGE_PATCH_TOKEN * SHIKRA_IMAGE_TOKEN_LENGTH
    qu = [template.replace("<question>", q) for q in query]
    qu = [p.replace("<ImageHere>", replace_token) for p in qu]

    input_tokens = tokenizer(
        qu, return_tensors="pt", padding="longest", add_special_tokens=False
    ).to("cuda")

    bs = len(query)
    bos = torch.ones([bs, 1], dtype=torch.int64, device="cuda") * tokenizer.bos_token_id
    input_ids = torch.cat([bos, input_tokens.input_ids], dim=1)

    img_start_idx = torch.where(input_ids == SHIKRA_IMG_START_TOKEN)[1][0].item()
    img_end_idx = torch.where(input_ids == SHIKRA_IMG_END_TOKEN)[1][0].item()

    kwargs = {}
    kwargs["input_ids"] = input_ids
    kwargs["images"] = image_tensor.to("cuda")

    return qu, img_start_idx, img_end_idx, kwargs


# Example usage:
# prepare_inputs_for_model(args, image, model, tokenizer, kwargs)


class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.vlm_model = None
        self.llm_model = None
        self.image_processor = None
        self.load_model()

    def load_model(self):
        if self.model_name == "llava-1.5":
            model_path = os.path.expanduser("/path/to/models/llava-v1.5-7b")
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_llava_model(model_path)
            )

        elif self.model_name == "minigpt4":
            cfg_path = "./minigpt4/eval_config/minigpt4_eval.yaml"
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_minigpt4_model(cfg_path)
            )

        elif self.model_name == "shikra":
            yaml_path = "./mllm/config/config.yml"
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_shikra_model(yaml_path)
            )

        else:
            raise ValueError(f"Unknown model: {self.model}")

    def prepare_inputs_for_model(self, template, query, image):
        if self.model_name == "llava-1.5":
            questions, img_start_idx, img_end_idx, kwargs = prepare_llava_inputs(
                template, query, image, self.tokenizer
            )
        elif self.model_name == "minigpt4":
            questions, img_start_idx, img_end_idx, kwargs = prepare_minigpt4_inputs(
                template, query, image, self.vlm_model
            )
        elif self.model_name == "shikra":
            questions, img_start_idx, img_end_idx, kwargs = prepare_shikra_inputs(
                template, query, image, self.tokenizer
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        self.img_start_idx = img_start_idx
        self.img_end_idx = img_end_idx

        return questions, kwargs

    def init_cfg_processor(self, questions, gamma=1.1, beam=1, start_layer=0, end_layer=32):
        if self.model_name == "minigpt4":
            chunks = [q.split("<Img><ImageHere></Img>") for q in questions]
        elif self.model_name == "llava-1.5":
            chunks = [q.split("<ImageHere>") for q in questions]
        elif self.model_name == "shikra":
            split_token = (
                "<im_start>"
                + DEFAULT_IMAGE_PATCH_TOKEN * SHIKRA_IMAGE_TOKEN_LENGTH
                + "<im_end>"
            )
            chunks = [q.split(split_token) for q in questions]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        chunk_before = [chunk[0] for chunk in chunks]
        chunk_after = [chunk[1] for chunk in chunks]

        token_before = self.tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).input_ids.to("cuda")
        token_after = self.tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).input_ids.to("cuda")

        batch_size = len(questions)
        bos = (
            torch.ones(
                [batch_size, 1], dtype=token_before.dtype, device=token_before.device
            )
            * self.tokenizer.bos_token_id
        )
        neg_promt = torch.cat([bos, token_before, token_after], dim=1)
        neg_promt = neg_promt.repeat(beam, 1)
        logits_processor = CFGLogits(gamma, neg_promt.to("cuda"), self.llm_model, start_layer=start_layer, end_layer=end_layer)

        return logits_processor

    def decode(self, output_ids):
        # get outputs
        if self.model_name == "llava-1.5":
            # replace image token by pad token
            output_ids = output_ids.clone()
            output_ids[output_ids == IMAGE_TOKEN_INDEX] = torch.tensor(
                0, dtype=output_ids.dtype, device=output_ids.device
            )

            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [text.split("ASSISTANT:")[-1].strip() for text in output_text]

        elif self.model_name == "minigpt4":
            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [
                text.split("###")[0].split("Assistant:")[-1].strip()
                for text in output_text
            ]

        elif self.model_name == "shikra":
            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [text.split("ASSISTANT:")[-1].strip() for text in output_text]

        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        return output_text
