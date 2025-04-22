# Ref: https://github.com/voidism/DoLa
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria
import numpy as np

from attention import llama_modify, llama_restore
import string
from nltk import pos_tag
import re

personal_pronouns = {
    # 主格
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    # 宾格
    'me', 'him', 'her', 'us', 'them'
}
possessive_pronouns = {
    # 形容词性物主代词
    'my', 'your', 'his', 'her', 'its', 'our', 'their',
    # 名词性物主代词
    'mine', 'yours', 'hers', 'ours', 'theirs'
}
be_verbs = {
    'be', 'am', 'is', 'are', 'was', 'were', 
    'been', 'being'
}
all_words = (
    personal_pronouns | 
    possessive_pronouns | 
    be_verbs | 
    {'following', 'did', 'do', 'does', 'according', 'q', 'a', 'which'}  # 保留原有其他词
)

def token_set(tagged):
    punctuation_set = []
    concept_n = []
    concept_adj = []
    concept_v = []
    no_concept = []
    ###
    
    for i, (word_ori, tag) in enumerate(tagged):
        word = re.sub(r'^[\\\'"\-]+', '', word_ori)
        word = word.lower()
        #print('w',word)
        if word in ['<s>', None, '"', ':', '.', ',', '?', '<0x0a>', ''] or word.startswith('_'):
            #print('p')
            punctuation_set.append(i)
            #token_word['pword'].add(word)
        elif tag in ['IN', 'DT', 'TO', 'MD', 'CC'] or tag.startswith('W') or word in all_words:
            no_concept.append(i)
            #token_word['ncword'].add(word)
        # else:
        #     concept_n.append(i)
        elif (tag.startswith('N') and word != 'which') or tag in ['CD', 'PRP', 'PRP$', 'POS']:
            concept_n.append(i)
            #token_word['cnword'].add(word)
        elif tag.startswith('J'):
            concept_adj.append(i)
            #token_word['caword'].add(word)
        elif tag.startswith('V'):
            concept_v.append(i)
            #token_word['cvword'].add(word)
    return punctuation_set, concept_n, concept_adj, no_concept, concept_v

def get_token_indices(models,tokens):
    model_type = models.lower()
    if 'llama' in model_type:
        start_sym = '▁'
    if 'qwen' in model_type:
        start_sym = 'Ġ'
    """ 根据 tokens 生成词性分类对应的原始 token 索引 """
    # 辅助函数：标点符号判断
    punctuation = set(string.punctuation)
    def is_punct(tok):
        return all(c in punctuation for c in tok)

    # 步骤 1: 合并子词为完整单词，记录原始索引
    current_word, current_indices = None, []
    words, columns_list = [], []
    
    for i, token in enumerate(tokens):
        # 预计算重要特征
        is_special = token in ['<0x0A>', '<s>']
        stripped_token = token.lstrip(start_sym) if start_sym else token
        is_stripped_punct = is_punct(stripped_token)
        has_start_sym = token.startswith(start_sym) if start_sym else False

        # 条件 1：非start_sym开头的特殊符号或标点（必须单独成词）
        if not has_start_sym and (is_special or is_punct(token)):
            # 中断当前词组合（如果有）
            if current_word is not None:
                words.append(current_word)
                columns_list.append(current_indices)
                current_word, current_indices = None, []
            # 单独成词（即使为空）
            words.append(token)
            columns_list.append([i])
            current_word, current_indices = None, []
            continue

        # 条件 2：start_sym开头且去前缀后是标点（必须单独成词）
        if has_start_sym and is_stripped_punct:
            # 中断当前词组合（如果有）
            if current_word is not None:
                words.append(current_word)
                columns_list.append(current_indices)
                current_word, current_indices = None, []
            # 单独成词
            words.append(stripped_token)
            columns_list.append([i])
            current_word, current_indices = None, []
            continue

        # 条件 3：start_sym开头且不是标点（开始新词）
        if has_start_sym:
            # 中断当前词组合（如果有）
            if current_word is not None:
                words.append(current_word)
                columns_list.append(current_indices)
            # 开始新词
            current_word = stripped_token
            current_indices = [i]
            continue

        # 条件 4：普通子词（需要组合）
        # 如果当前没有正在组合的词，说明出现非法情况（非start_sym开头的普通token）
        if current_word is None:
            current_word = token  # 或者根据需求处理错误
            current_indices = [i]
        else:
            current_word += token
            current_indices.append(i)

    # 处理最后一个未完成的词
    if current_word is not None:
        words.append(current_word)
        columns_list.append(current_indices)

    if not words:
        return None
    # 步骤 2: 词性标注与分类
    #print(words)
    tagged = pos_tag(words)
    
    # # 简化版分类逻辑（需根据实际 token_set 实现调整）
    # p_indices, cn_indices, ca_indices, cv_indices, nc_indices = [], [], [], [], []
    # for idx, (word, tag) in enumerate(tagged):
    #     word_clean = re.sub(r'^[\\\'"\-]+', '', word)
    #     if word_clean in {'<s>', '<0x0A>'} or is_punct(word):
    #         p_indices.extend(columns_list[idx])
    #     elif tag.startswith('N') or tag in ['CD', 'PRP']:
    #         cn_indices.extend(columns_list[idx])
    #     elif tag.startswith('J'):
    #         ca_indices.extend(columns_list[idx])
    #     elif tag.startswith('V'):
    #         cv_indices.extend(columns_list[idx])
    #     else:
    #         nc_indices.extend(columns_list[idx])
            
    #p_indices, cn_indices, ca_indices, nc_indices, cv_indices = token_set(tagged)
    #print('incl',p_indices, cn_indices, ca_indices, nc_indices, cv_indices)
    punctuation_set, concept_n, concept_adj, no_concept, concept_v = token_set(tagged)
    #print('attn',punctuation_set, concept_n, concept_adj, no_concept, concept_v)
    
    # 对应token标签汇聚 便于整体操作
    if punctuation_set:
        ### token idx
        p_idx = [num for idx in punctuation_set for num in columns_list[idx]]
    else:
        p_idx = []
        
    if concept_n:
        cn_idx = [num for idx in concept_n for num in columns_list[idx]]
    else:
        cn_idx = []
        
    if concept_adj:
        ca_idx = [num for idx in concept_adj for num in columns_list[idx]]
    else:
        ca_idx = []
        
    if no_concept:
        nc_idx = [num for idx in no_concept for num in columns_list[idx]]
    else:
        nc_idx = []
        
    if concept_v:
        cv_idx = [num for idx in concept_v for num in columns_list[idx]]
    else:
        cv_idx = []
            
    if 0 not in p_idx:
        p_idx.append(0)
    
    for ind in [cn_idx, ca_idx, cv_idx, nc_idx]:
        if ind is not None and 0 in ind:
            ind[:] = [x for x in ind if x !=0]

    return p_idx, cn_idx, ca_idx, cv_idx, nc_idx

class SLED_DecodedLLM_TruthfulQA:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)

        self.num_layers = self.model.config.num_hidden_layers if hasattr(self.model.config,
                                                                         "num_hidden_layers") else self.model.config.n_layer

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     low_cpu_mem_usage=False, attn_implementation="eager", **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()

        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1,
                                min_tokens_to_keep: int = 1):

        scores_normalized = scores.log_softmax(dim=-1)

        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)

        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]

        probs_max = torch.max(scores_normalized, dim=-1).values

        probs_thresh = probs_max + np.log(relative_top)

        probs_thresh = torch.min(min_thresh, probs_thresh)

        probs_thresh = probs_thresh.unsqueeze(-1)

        return scores_normalized < probs_thresh

    def lm_score(self, input_text1, input_text2, start_layer, end_layer, attn_alpha, token_enhance, token_weaken, alpha, pmi=False,
                 mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='VanillaGreedy',
                 verbose=True,
                 remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True,
                 evolution_rate=2, evolution_scale=10, evolution_lower_bound=-100, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_encoding = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            prefix_encoding = self.tokenizer(input_text1, return_tensors="pt").to(self.device)
            input_ids = input_encoding['input_ids']
            prefix_ids = prefix_encoding['input_ids']
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            attention_mask = input_encoding['attention_mask']

            if mode == 'VanillaGreedy':
                outputs = self.model(input_ids)[0].squeeze(0)
                if post_softmax:
                    outputs = outputs.log_softmax(-1)
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            elif mode == 'attn':
                outputs = self.model(input_ids)[0].squeeze(0)
                #if post_softmax:
                outputs = outputs.log_softmax(-1)
                outputs_ori = outputs[prefix_ids.shape[-1] - 1: -1, :]
                log_probs = outputs_ori[range(outputs_ori.shape[0]), continue_ids].sum().item()
                
                ###cd
                query_tokens = input_ids.tolist()
                #print(query_tokens, prompt_ids.tolist())
                query_text = self.tokenizer.convert_ids_to_tokens(query_tokens[0], skip_special_tokens=False)
                p_idx, cn_idx, ca_idx, cv_idx, nc_idx = get_token_indices('llama', query_text)
                if token_enhance == 'cn':
                    token_enhance_idx = cn_idx
                elif token_enhance == 'ac':
                    token_enhance_idx = cn_idx + ca_idx + cv_idx
                elif token_enhance == 'nc':
                    token_enhance_idx = nc_idx
                else:
                    token_enhance_idx = None
                    
                if token_weaken == 'cn':
                    token_weaken_idx = cn_idx
                elif token_weaken == 'ac':
                    token_weaken_idx = cn_idx + ca_idx + cv_idx
                elif token_weaken == 'nc':
                    token_weaken_idx = nc_idx
                elif token_weaken == 'p':
                    token_weaken_idx = p_idx
                else:
                    token_weaken_idx = None

                llama_modify(
                    self.model,
                    'llama',
                    start_layer=start_layer,
                    end_layer=end_layer,
                    use_attn=True,
                    alpha=attn_alpha,
                    first_token_idx=prefix_ids.shape[-1] - 1,
                    token_enhance=token_enhance_idx,
                    token_weaken=token_weaken_idx
                )
                
                outputs_modified =self.model(input_ids)[0].squeeze(0)
                #print(outputs_modified)
                outputs_modified = outputs_modified.log_softmax(-1)
                outputs_modified = outputs_modified[prefix_ids.shape[-1] - 1: -1, :]
                outputs_cd = outputs_ori - outputs_modified
                if post_softmax:
                    outputs_cd = outputs_cd.log_softmax(dim=-1)
                # get logprobs for each token in the answer
                log_probs = outputs_cd[range(outputs_cd.shape[0]), continue_ids].sum().item()
                #print('ori', outputs_ori[range(outputs_ori.shape[0]), continue_ids].sum().item(), 'after', outputs_modified[range(outputs_modified.shape[0]), continue_ids].sum().item())
                llama_restore(self.model, 'llama', start_layer=2, end_layer=16)
                
            elif mode == 'dola':
                premature_layer_dist = {l: 0 for l in candidate_premature_layers}
                premature_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack(
                        [dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :],
                                                     dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers,
                                                         dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :,
                               :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :],
                                                             dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers,
                                                                 dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(
                        -1)  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(
                        -1)  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer)

                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(premature_layers):
                    base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)

                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()


            elif mode == 'SLED':
                premature_layer_dist = {l: 0 for l in candidate_premature_layers}
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )
                new_output_logits = dict_outputs[mature_layer].clone()

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    stacked_premature_layers = torch.stack(
                        [dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :],
                                                     dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers,
                                                         dim=-1)
                    topk_prob, topk_indices = torch.topk(softmax_mature_layer, evolution_scale)
                    topk_indices = topk_indices[0]

                    divergence = stacked_premature_layers - dict_outputs[mature_layer][:, seq_i, :]
                    candidate_gradients_expanded = softmax_premature_layers.expand(-1, len(topk_indices), -1)
                    candidate_mask = torch.zeros_like(candidate_gradients_expanded)
                    topk_indices_expanded = topk_indices.unsqueeze(0).unsqueeze(2)
                    candidate_mask.scatter_(2, topk_indices_expanded.expand(softmax_premature_layers.size(0), -1, -1),
                                            1)
                    candidate_gradients_expanded = candidate_gradients_expanded - candidate_mask
                    candidate_gradients_expanded = candidate_gradients_expanded.to(torch.float32)
                    layer_divergence_expanded = divergence.to(torch.float32)

                    layer_dot_results = F.cosine_similarity(candidate_gradients_expanded, layer_divergence_expanded,
                                                            dim=2)
                    layer_topk_values, layer_topk_indices = torch.topk(layer_dot_results, evolution_scale)
                    layer_topk_topk_indices = topk_indices[layer_topk_indices]

                    layer_topk_values = (layer_topk_values * (layer_topk_values > 0)) ** 2
                    layer_topk_values_sum_layers = torch.sum(layer_topk_values, dim=1).clone()
                    non_zero_indices = layer_topk_values_sum_layers != 0
                    layer_topk_values[non_zero_indices] /= layer_topk_values_sum_layers[non_zero_indices].unsqueeze(1)
                    if layer_topk_values_sum_layers.sum() != 0:
                        layer_topk_values_sum_layers = layer_topk_values_sum_layers / layer_topk_values_sum_layers.sum()
                    proxy_gradients_tensor_delta = torch.zeros_like(softmax_mature_layer,
                                                                    device=layer_divergence_expanded.device).to(
                        layer_divergence_expanded.dtype).repeat(layer_topk_values.size(0), 1)
                    proxy_gradients_tensor_delta.scatter_(1, layer_topk_topk_indices, -layer_topk_values)
                    proxy_gradients_tensor_delta = torch.sum(
                        proxy_gradients_tensor_delta * layer_topk_values_sum_layers.unsqueeze(1), dim=0)
                    proxy_gradients_tensor_delta = proxy_gradients_tensor_delta.to(softmax_mature_layer.dtype)
                    hidden_states_seq_i = new_output_logits[:, seq_i, :].clone()

                    op_T = 10
                    evolution_rate_values = [evolution_rate * (1 - i / op_T) for i in range(op_T)]

                    for op_t in range(op_T):
                        lr_t = evolution_rate_values[op_t]
                        softmax_hidden_states_seq_i = F.softmax(hidden_states_seq_i, dim=-1)
                        proxy_gradients_tensor = softmax_hidden_states_seq_i + proxy_gradients_tensor_delta
                        hidden_states_seq_i.sub_(lr_t * proxy_gradients_tensor)

                    hidden_states_seq_i_new = torch.full_like(hidden_states_seq_i[0], fill_value=evolution_lower_bound,
                                                              device=hidden_states_seq_i.device,
                                                              dtype=hidden_states_seq_i.dtype)
                    hidden_states_seq_i_new[topk_indices] = hidden_states_seq_i[0, topk_indices]
                    new_output_logits[:, seq_i, :] = hidden_states_seq_i_new.unsqueeze(dim=0)

                if post_softmax:
                    log_new_output_logits = F.log_softmax(new_output_logits, dim=-1)
                else:
                    log_new_output_logits = new_output_logits

                log_new_output_logits = log_new_output_logits[0, prefix_ids.shape[-1] - 1: -1, :]
                log_probs = log_new_output_logits[range(log_new_output_logits.shape[0]), continue_ids].sum().item()

        return log_probs, (premature_layer_dist if mode == 'dola' else None)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1,
                                min_tokens_to_keep: int = 1):

        scores_normalized = scores.log_softmax(dim=-1)

        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)

        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]

        probs_max = torch.max(scores_normalized, dim=-1).values

        probs_thresh = probs_max + np.log(relative_top)

        probs_thresh = torch.min(min_thresh, probs_thresh)

        probs_thresh = probs_thresh.unsqueeze(-1)

        return scores_normalized < probs_thresh