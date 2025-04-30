# Ref: https://github.com/voidism/DoLa
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria
import numpy as np
from transformers.generation.logits_process import LogitsProcessorList
from attention import llama_modify, llama_restore
from attention_adaptive import llama_modify_adaptive, llama_restore_adaptive
import string
from nltk import pos_tag
import re
from CFG import CFGLogits
import random

def advanced_sample(lst, percent, seed=None, keep_order=False):
    if seed is not None:
        random.seed(seed)
    
    X = max(0, min(100, percent))
    n = len(lst)
    keep_num = round(n * X)
    
    if keep_order:
        # 保留原始顺序的随机选择
        indices = sorted(random.sample(range(n), keep_num))
        return [lst[i] for i in indices]
    else:
        return random.sample(lst, keep_num)

def split_llama_special(token, sep='Ċ'):
    # 查找最后一个sep的位置
    last_idx = token.find(sep)
    if last_idx == -1:
        return token
    return token[:last_idx]

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
    {'following', 'did', 'do', 'does', 'according', 'which','q','a'}  # 保留原有其他词
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
        #print(word)
        #print('w',word)
        if word in ['<|begin_of_text|>','<s>', None, '"', ':', '.', ',', '?', '<0x0a>', ''] or word.startswith('_') or word.startswith('Ġ'):
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
    if 'llama3' in model_type:
        start_sym = 'Ġ'
    elif 'llama' in model_type:
        start_sym = '▁'
    elif 'qwen' in model_type:
        start_sym = 'Ġ'
    is_llama3 = 'llama3' in model_type
    # 辅助函数：标点符号判断
    punctuation = set(string.punctuation)
    def is_punct(tok):
        return all(c in punctuation for c in tok)

    # 步骤 1: 合并子词为完整单词，记录原始索引
    current_word, current_indices = None, []
    words, columns_list = [], []
    
    force_new_word = False
    for i, token in enumerate(tokens):
        #print('token', token)
        if force_new_word:
            if current_word is not None:
                words.append(current_word)
                columns_list.append(current_indices)
                current_word, current_indices = None, []
            # 无论是否有前缀，强制作为新词开头
            stripped_part = token.lstrip(start_sym) if start_sym else token
            current_word = stripped_part
            current_indices = [i]
            force_new_word = False  # 重置标志
            continue
        # 预计算重要特征
        is_special = token in ['<0x0A>', '<s>', '<|begin_of_text|>']
        stripped_token = token.lstrip(start_sym) if start_sym else token
        is_stripped_punct = is_punct(stripped_token)
        has_start_sym = token.startswith(start_sym) if start_sym else False

        # ===== 条件1修改 =====
        if not has_start_sym and (is_special or is_punct(token) or is_punct(token[0])):
            if is_llama3 and not is_special:
                # 分割到最后一个Ċ前的内容
                head_part = split_llama_special(token)  # 直接丢弃剩余部分
                # 中断当前词并保存头部
                if current_word is not None:
                    words.append(current_word)
                    columns_list.append(current_indices)
                    current_word, current_indices = None, []
                #print('head',head_part)
                words.append(head_part)
                columns_list.append([i])
                # 标记下一个token必须为新词
                force_new_word = True
                continue
            else:
                if current_word is not None:
                    words.append(current_word)
                    columns_list.append(current_indices)
                    current_word, current_indices = None, []
                words.append(token)
                columns_list.append([i])
                current_word, current_indices = None, []
                continue

        # ===== 条件2：带前缀的标点处理 =====
        if has_start_sym and is_stripped_punct:
            if is_llama3:
                # 分割并丢弃剩余部分
                stripped_head = split_llama_special(stripped_token)
                # 保存处理后的标头
                if current_word is not None:
                    words.append(current_word)
                    columns_list.append(current_indices)
                words.append(stripped_head)
                columns_list.append([i])
                # 标记下一个token必须为新词
                force_new_word = True
                continue
            else:
                # 原有逻辑
                if current_word is not None:
                    words.append(current_word)
                    columns_list.append(current_indices)
                    current_word, current_indices = None, []
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
    punctuation_set, concept_n, concept_adj, no_concept, concept_v = token_set(tagged)
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
                                                    low_cpu_mem_usage=True, attn_implementation="eager", **kwargs)

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

    def lm_score(self, input_text1, input_text2, demo, question, start_layer, end_layer, attn_alpha, token_enhance, token_weaken, beta, sink, sink_layers,ema, th,pmi=False,
                mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='VanillaGreedy',
                verbose=True,
                remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=False,
                evolution_rate=2, evolution_scale=10, evolution_lower_bound=-100, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            demo_encoding = self.tokenizer(demo, return_tensors="pt").to(self.device)
            question_encoding = self.tokenizer(question, return_tensors="pt").to(self.device)
            demo_ids = demo_encoding['input_ids']
            question_ids = question_encoding['input_ids']
            
            input_encoding = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            prefix_encoding = self.tokenizer(input_text1, return_tensors="pt").to(self.device)
            input_ids = input_encoding['input_ids']
            prefix_ids = prefix_encoding['input_ids']
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            attention_mask = input_encoding['attention_mask']

            if mode == 'VanillaGreedy':
                #print(input_ids)
                outputs = self.model(input_ids)[0].squeeze(0)
                #print('qwen',outputs)
                #print(outputs.size())
                if post_softmax:
                    outputs = outputs.log_softmax(-1)
                    
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]
                #print(outputs)
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            elif mode == 'attn':
                outputs = self.model(input_ids)[0].squeeze(0)
                # if post_softmax:
                #     outputs = outputs.log_softmax(-1)
                outputs_ori = outputs[prefix_ids.shape[-1] - 1: -1, :]
                
                ###cd
                query_tokens = input_ids.tolist()
                question_tokens = demo_ids.tolist()
                pre_num = len(question_ids[0])
                question_tokens = question_tokens[0][question_ids.shape[-1]:]

                question_text = self.tokenizer.convert_ids_to_tokens(question_tokens, skip_special_tokens=False)
                query_text = self.tokenizer.convert_ids_to_tokens(query_tokens[0], skip_special_tokens=False)
                #print(query_text,question_text)
                p_idx, cn_idx, ca_idx, cv_idx, nc_idx = get_token_indices('llama', query_text)
                ac_idx = cn_idx + ca_idx + cv_idx
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
                    
                # token_weaken_idx = advanced_sample(token_weaken_idx, 0.5, seed=10, keep_order=True)
                ### Ori Weaken
                # llama_modify(
                #     self.model,
                #     'llama',
                #     start_layer=start_layer,
                #     end_layer=end_layer,
                #     use_attn=True,
                #     alpha=attn_alpha,
                #     first_token_idx=prefix_ids.shape[-1] - 1,
                #     token_enhance=token_enhance_idx,
                #     token_weaken=token_weaken_idx,
                #     sink = sink,
                #     special_layers=sink_layers
                # )
                llama_modify_adaptive(
                    self.model,
                    'llama',
                    start_layer=start_layer,
                    end_layer=end_layer,
                    use_attn=True,
                    alpha=attn_alpha,
                    first_token_idx=prefix_ids.shape[-1] - 1,
                    token_enhance=token_enhance_idx,
                    token_weaken=token_weaken_idx,
                    n_idx=nc_idx,
                    c_idx=ac_idx,
                    s_idx=p_idx,
                    th=th,
                    ave_token= torch.zeros((1, outputs.shape[0] - prefix_ids.shape[-1]+1), dtype=torch.float32), # C,P,N
                    sink = sink,
                    ema = ema,
                    special_layers=sink_layers
                )
                
                outputs_modified =self.model(input_ids)[0].squeeze(0)
                #print(outputs_modified)
                ###new 
                outputs_log = outputs.log_softmax(-1)
                outputs_ori_log = outputs_log[prefix_ids.shape[-1] - 1: -1, :]
                
                outputs_modified_log = outputs_modified.log_softmax(-1)
                outputs_modified_log = outputs_modified_log[prefix_ids.shape[-1] - 1: -1, :]
                
                outputs_modified = outputs_modified[prefix_ids.shape[-1] - 1: -1, :]
                
                #outputs_cd = outputs_ori + attn_alpha*(outputs_ori_log - outputs_modified)
                #outputs_cd = beta*(outputs_ori) + (1-beta)*(outputs_modified_log)
                #outputs_cd = (1-attn_alpha)*outputs_ori + attn_alpha* (outputs_ori - outputs_modified)
                #outputs_cd = outputs_modified_log
                outputs_cd = beta*outputs_ori_log +(outputs_ori_log- outputs_modified_log)
                if post_softmax:
                    outputs_cd = outputs_cd.log_softmax(dim=-1)
                # if relative_top > 0.0:
                #     relative_top_mask = self.get_relative_top_filter(outputs_ori, relative_top)
                #     outputs_cd = torch.where(relative_top_mask, relative_top_value, outputs_cd)

                log_probs = outputs_cd[range(outputs_cd.shape[0]), continue_ids].sum().item()
                llama_restore_adaptive(self.model, 'llama', start_layer=0, end_layer=len(self.model.model.layers)-1)
                
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
    
class SLED_DecodedLLM_Factor:
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
            kwargs = {"torch_dtype": torch.float32, "offload_folder": f"{model_name}/offload"}
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

        # tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        # model = AutoModelForCausalLM.from_pretrained(model_name,
        #                                             low_cpu_mem_usage=True, attn_implementation="eager", **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    low_cpu_mem_usage=True, attn_implementation="eager", **kwargs)
    
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

    def lm_score(self, input_text1, input_text2, start_layer, end_layer, attn_alpha, token_enhance, token_weaken, beta, sink,sink_layers,th,ema, pmi=False,
                mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='VanillaGreedy',
                verbose=True,
                remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True,
                evolution_rate=2, evolution_scale=10, evolution_lower_bound=-2500, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]

            if mode == 'VanillaGreedy':
                outputs = self.model(input_ids)[0].squeeze(0)
                if post_softmax:
                    outputs = outputs.log_softmax(-1)
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            elif mode == 'attn':
                outputs = self.model(input_ids)[0].squeeze(0)
                # if post_softmax:
                #     outputs = outputs.log_softmax(-1)
                outputs_ori = outputs[prefix_ids.shape[-1] - 1: -1, :]
                
                ###cd
                query_tokens = input_ids.tolist()
                #print(query_tokens, prompt_ids.tolist())
                query_text = self.tokenizer.convert_ids_to_tokens(query_tokens[0], skip_special_tokens=False)
                p_idx, cn_idx, ca_idx, cv_idx, nc_idx = get_token_indices('llama', query_text)
                ac_idx = cn_idx + ca_idx + cv_idx
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
                    
                token_weaken_idx = advanced_sample(token_weaken_idx, 0.5, seed=10, keep_order=True)
                #token_enhance_idx = advanced_sample(token_enhance_idx, 0.5, seed=10, keep_order=True)
                ###ori mod
                # llama_modify(
                #     self.model,
                #     'llama',
                #     start_layer=start_layer,
                #     end_layer=end_layer,
                #     use_attn=True,
                #     alpha=attn_alpha,
                #     first_token_idx=prefix_ids.shape[-1] - 1,
                #     token_enhance=token_enhance_idx,
                #     token_weaken=token_weaken_idx,
                #     sink = sink,
                #     special_layers=sink_layers
                # )
                llama_modify_adaptive(
                    self.model,
                    'llama',
                    start_layer=start_layer,
                    end_layer=end_layer,
                    use_attn=True,
                    alpha=attn_alpha,
                    first_token_idx=prefix_ids.shape[-1] - 1,
                    token_enhance=token_enhance_idx,
                    token_weaken=token_weaken_idx,
                    n_idx=nc_idx,
                    c_idx=ac_idx,
                    s_idx=p_idx,
                    th=th,
                    ave_token= torch.zeros((1, outputs.shape[0] - prefix_ids.shape[-1]+1), dtype=torch.float32), # C
                    sink = sink,
                    ema = ema,
                    special_layers=sink_layers
                )
                
                outputs_modified =self.model(input_ids)[0].squeeze(0)
                #print(outputs_modified)
                ###new 
                outputs_log = outputs.log_softmax(-1)
                outputs_ori_log = outputs_log[prefix_ids.shape[-1] - 1: -1, :]
                
                outputs_modified_log = outputs_modified.log_softmax(-1)
                outputs_modified_log = outputs_modified_log[prefix_ids.shape[-1] - 1: -1, :]
                
                outputs_modified = outputs_modified[prefix_ids.shape[-1] - 1: -1, :]
                #outputs_cd = beta*outputs_ori_log +(outputs_ori_log- outputs_modified_log)
                outputs_cd=outputs_modified_log
                if post_softmax:
                    outputs_cd = outputs_cd.log_softmax(dim=-1)
                # if relative_top > 0.0:
                #     relative_top_mask = self.get_relative_top_filter(outputs_ori, relative_top)
                #     outputs_cd = torch.where(relative_top_mask, relative_top_value, outputs_cd)

                log_probs = outputs_cd[range(outputs_cd.shape[0]), continue_ids].sum().item()
                llama_restore_adaptive(self.model, 'llama', start_layer=0, end_layer=len(self.model.model.layers)-1)


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

                    op_T = 1
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
    
    
class SLED_DecodedLLM_MMLU:
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
        ### attn calculate method
        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    low_cpu_mem_usage=True, attn_implementation="eager", **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()

        return model, tokenizer

    # def set_stop_words(self, stop_words):
    #     self.stop_words = stop_words
    #     self.stopping_criteria = StoppingCriteriaList()
    #     list_stop_word_ids = []
    #     for stop_word in self.stop_words:
    #         stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
    #         list_stop_word_ids.append(stop_word_ids)
    #         print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
    #     self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

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

    def lm_score(self, input_text1, input_text2, start_layer, end_layer, attn_alpha, token_enhance, token_weaken, beta, sink, pmi=False,
                mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='VanillaGreedy',
                verbose=True,
                remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True,
                evolution_rate=2, evolution_scale=10, evolution_lower_bound=-2500, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]

            if mode == 'VanillaGreedy':
                outputs = self.model(input_ids)[0].squeeze(0)
                if post_softmax:
                    outputs = outputs.log_softmax(-1)
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            elif mode == 'attn':
                #print('input_ids',input_ids)
                #print('input_ids',input_ids)
                outputs = self.model(input_ids)[0].squeeze(0)
                # if post_softmax:
                #     outputs = outputs.log_softmax(-1)
                outputs_ori = outputs[prefix_ids.shape[-1] - 1: -1, :]
                
                ###cd
                query_tokens = input_ids.tolist()
                #print(query_tokens, prompt_ids.tolist())
                query_text = self.tokenizer.convert_ids_to_tokens(query_tokens[0], skip_special_tokens=False)
                # print(len(query_text))
                # for ii in [234, 235, 236, 241, 242]:
                #     print(query_text[ii])
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
                #print('idx',token_weaken_idx)
                llama_modify(
                    self.model,
                    'llama',
                    start_layer=start_layer,
                    end_layer=end_layer,
                    use_attn=True,
                    alpha=attn_alpha,
                    first_token_idx=prefix_ids.shape[-1] - 1,
                    token_enhance=token_enhance_idx,
                    token_weaken=token_weaken_idx,
                    sink = sink
                )
                
                outputs_modified =self.model(input_ids)[0].squeeze(0)
                #print(outputs_modified)
                ###new 
                outputs_log = outputs.log_softmax(-1)
                outputs_ori_log = outputs_log[prefix_ids.shape[-1] - 1: -1, :]
                #print('outputs_ori_log', outputs_ori_log[range(outputs_ori_log.shape[0]), continue_ids].sum().item())
                outputs_modified_log = outputs_modified.log_softmax(-1)
                outputs_modified_log = outputs_modified_log[prefix_ids.shape[-1] - 1: -1, :]
                
                outputs_modified = outputs_modified[prefix_ids.shape[-1] - 1: -1, :]
                # print('ans_id', continue_ids)
                # print('ori_modified', outputs_modified_log)
                # print('modified', outputs_modified_log[range(outputs_modified_log.shape[0]), continue_ids].sum().item())
                #outputs_cd = outputs_ori + attn_alpha*(outputs_ori_log - outputs_modified)
                #outputs_cd = beta*(outputs_ori) + (1-beta)*(outputs_modified_log)
                #outputs_cd = (1-attn_alpha)*outputs_ori + attn_alpha* (outputs_ori - outputs_modified)
                #outputs_cd = outputs_modified_log
                outputs_cd = beta*outputs_ori_log +(outputs_ori_log- outputs_modified_log)
                #print('outputs_ori_log',outputs_ori_log,'outputs_modified_log',outputs_modified_log)
                if post_softmax:
                    outputs_cd = outputs_cd.log_softmax(dim=-1)
                # if relative_top > 0.0:
                #     relative_top_mask = self.get_relative_top_filter(outputs_ori, relative_top)
                #     outputs_cd = torch.where(relative_top_mask, relative_top_value, outputs_cd)

                log_probs = outputs_cd[range(outputs_cd.shape[0]), continue_ids].sum().item()
                #print('log_probs',log_probs)
                llama_restore(self.model, 'llama', start_layer=start_layer, end_layer=end_layer)


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

                    op_T = 1
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
