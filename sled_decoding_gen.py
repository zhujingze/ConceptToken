# Ref: https://github.com/voidism/DoLa
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria
import numpy as np
from transformers.generation.logits_process import LogitsProcessorList
from attention_gen_adaptive import llama_modify_adaptive
import string
from nltk import pos_tag
import re
from CFG import CFGLogits

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
    
class SLED_DecodedLLM_StrQA:
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
        
        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')

        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     low_cpu_mem_usage=True,  attn_implementation="eager",**kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()

        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            if 'llama3' in self.model_name:
                if stop_word == "Q:":
                    stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
                else:
                    stop_word_ids = self.tokenizer.encode(stop_word)[1:]

            else:
                stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))



    def generate(self, input_text, sink_layers,sink,beta,token_weaken,
                token_enhance,attn_alpha,start_layer,end_layer, th, ema, including_answers,
                max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None,
                premature_layer=None, candidate_premature_layers=[], attn_layers=[], mode='VanillaGreedy', verbose=True,
                remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True,
                evolution_rate=2, evolution_scale=10, evolution_lower_bound=-1000, **kwargs):
        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'VanillaGreedy':
                with torch.inference_mode():
                    outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                                output_scores=True, return_dict_in_generate=True, dola_decoding=False,
                                                top_p=top_p, top_k=top_k, temperature=temperature,
                                                stopping_criteria=self.stopping_criteria, **kwargs)
                
            elif mode == 'attn':
                query_tokens = input_ids.tolist()
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
                
                # llama_modify(
                #     self.model,
                #     'llama',
                #     start_layer=start_layer,
                #     end_layer=end_layer,
                #     use_attn=True,
                #     use_cfg=False,
                #     alpha=attn_alpha,
                #     first_token_idx=input_ids.shape[-1] - 1,
                #     token_enhance=token_enhance_idx,
                #     token_weaken=token_weaken_idx,
                #     sink = sink,
                #     special_layers = sink_layers
                # )
                llama_modify_adaptive(
                    self.model,
                    self.tokenizer,
                    'llama',
                    start_layer=start_layer,
                    end_layer=end_layer,
                    use_attn=True,
                    use_cfg=False,
                    alpha=attn_alpha,
                    first_token_idx=input_ids.shape[-1] - 1,
                    token_enhance=token_enhance_idx,
                    token_weaken=token_weaken_idx,
                    sink = sink,
                    n_idx = nc_idx,
                    c_idx = ac_idx,
                    s_idx = p_idx,
                    th = th,
                    ema = ema,
                    ave_token = torch.zeros((1, 1), dtype=torch.float32),
                    special_layers = sink_layers
                )
                existing = set(sink_layers)
                new_layers = [x for x in range(start_layer, end_layer + 1) if x not in existing]
                # 合并原始列表与新元素
                attn_layers = sink_layers + new_layers
                # logits_processor = CFGLogits(beta, input_ids, self.model, start_layer=start_layer, end_layer=end_layer)
                # kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1, attn_layers = attn_layers,
                                              output_scores=True, return_dict_in_generate=True, dola_decoding=False, attn_decoding=True, beta=beta, including_answers=including_answers,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, **kwargs)

            elif mode == 'dola':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                              output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, relative_top=relative_top,
                                              mature_layer=mature_layer, premature_layer=None,
                                              candidate_premature_layers=candidate_premature_layers, **kwargs, )
                premature_layer_dist = outputs.premature_layer_dist


            elif mode == 'SLED':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                              output_scores=True, return_dict_in_generate=True,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, relative_top=relative_top,
                                              mature_layer=mature_layer, premature_layer=None,
                                              candidate_premature_layers=candidate_premature_layers,
                                              relative_top_value=relative_top_value, sled_decoding=True,
                                              evolution_rate=evolution_rate, evolution_scale=evolution_scale,
                                              evolution_lower_bound=evolution_lower_bound, **kwargs, )
                premature_layer_dist = outputs.premature_layer_dist

            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, (premature_layer_dist if mode == 'dola' else None)

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
    
    
class SLED_DecodedLLM_GSM8K:
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

        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')

        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     low_cpu_mem_usage=True, attn_implementation="eager",**kwargs)

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

    def generate(self, input_text, including_answers,th,ema,sink_layers,sink,beta,token_weaken,
        token_enhance,attn_alpha,start_layer,end_layer,
        max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None,
        premature_layer=None, candidate_premature_layers=[], attn_layers=[], mode='VanillaGreedy', verbose=True,
        remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True,
        evolution_rate=2, evolution_scale=10, evolution_lower_bound=-1000, **kwargs):

        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'VanillaGreedy':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                              output_scores=True, return_dict_in_generate=True, dola_decoding=False,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, **kwargs)
                
            elif mode == 'attn':
                query_tokens = input_ids.tolist()
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
                
                llama_modify_adaptive(
                    self.model,
                    self.tokenizer,
                    'llama',
                    start_layer=start_layer,
                    end_layer=end_layer,
                    use_attn=True,
                    use_cfg=False,
                    alpha=attn_alpha,
                    first_token_idx=input_ids.shape[-1] - 1,
                    token_enhance=token_enhance_idx,
                    token_weaken=token_weaken_idx,
                    sink = sink,
                    n_idx = nc_idx,
                    c_idx = ac_idx,
                    s_idx = p_idx,
                    th = th,
                    ema = ema,
                    ave_token = torch.zeros((1, 1), dtype=torch.float32),
                    special_layers = sink_layers
                )
                existing = set(sink_layers)
                new_layers = [x for x in range(start_layer, end_layer + 1) if x not in existing]
                # 合并原始列表与新元素
                attn_layers = sink_layers + new_layers
                # logits_processor = CFGLogits(beta, input_ids, self.model, start_layer=start_layer, end_layer=end_layer)
                # kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1, attn_layers = attn_layers,
                                              output_scores=True, return_dict_in_generate=True, dola_decoding=False, attn_decoding=True, beta=beta, including_answers=including_answers,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, **kwargs)
                

            elif mode == 'dola':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                              output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, relative_top=relative_top,
                                              mature_layer=mature_layer, premature_layer=None,
                                              candidate_premature_layers=candidate_premature_layers, **kwargs, )
                premature_layer_dist = outputs.premature_layer_dist


            elif mode == 'SLED':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                              output_scores=True, return_dict_in_generate=True,
                                              top_p=top_p, top_k=top_k, temperature=temperature,
                                              stopping_criteria=self.stopping_criteria, relative_top=relative_top,
                                              mature_layer=mature_layer, premature_layer=None,
                                              candidate_premature_layers=candidate_premature_layers,
                                              relative_top_value=relative_top_value, sled_decoding=True,
                                              evolution_rate=evolution_rate, evolution_scale=evolution_scale,
                                              evolution_lower_bound=evolution_lower_bound, **kwargs, )
                premature_layer_dist = outputs.premature_layer_dist

            sequences, scores = outputs.sequences, outputs.scores
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, (premature_layer_dist if mode == 'dola' else None)

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
