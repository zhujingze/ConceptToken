import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np
import string, re
from nltk import pos_tag
#import ipdb
import pickle
from torch.nn.functional import softmax
from attention import llama_modify, llama_restore
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
    punctuation_set, concept_n, concept_adj, no_concept, concept_v = token_set(tagged)
    
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

class Concept:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)
        

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
            low_cpu_mem_usage=True, **kwargs)

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

    # def generate(self, input_text,head_config, token_ranges=None,
    #              max_new_tokens=256, 
    #              top_p=0.95, 
    #              top_k=0,
    #              temperature=0.8, 
    #              task='halusum', 
    #              remove_stop_words=False, 
    #              alphas='2',
    #              scale=0.01,
    #              flag='None',
    #              **kwargs):
    #     with torch.no_grad():
    #         input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
    #         max_len = input_ids.shape[-1] + max_new_tokens
            
    #         outputs = self.model.generate(input_ids,max_length=max_len, num_return_sequences=1,
    #                             output_scores=True, return_dict_in_generate=True,top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria,**kwargs)
    #         induce_outputs = hicd_config(self.model,self.tokenizer,input_text,head_config,token_ranges=token_ranges,scale=scale,max_length=max_len,task=task,flag=flag)
            
    #         induce_sequences, induce_scores = induce_outputs.sequences, induce_outputs.scores
    #         sequences, scores = outputs.sequences, outputs.scores
            
            
    #         output_str_list =[]
    #         gen_probs = [score for score in scores]  
    #         induce_probs = [score for score in induce_scores]
            
    #         alphas = list(map(float, alphas.split(',')))
    #         for alpha in alphas:
    #             amplified_probs = [
    #                 softmax(gen_prob + alpha * (gen_prob - induce_prob),dim=-1)
    #                 for gen_prob, induce_prob in zip(gen_probs, induce_probs)
    #             ]

    #             amplified_tokens = [
    #                 torch.argmax(amplified_prob, dim=-1) for amplified_prob in amplified_probs
    #             ]

    #             amplified_sequences = torch.cat(amplified_tokens).cpu().numpy() if isinstance(amplified_tokens[0], torch.Tensor) else amplified_tokens
    #             amplified_output = self.tokenizer.decode(amplified_sequences, skip_special_tokens=True)
    #             output_str =amplified_output

    #             if remove_stop_words:
    #                 for stop_word in self.stop_words:
    #                     length_to_remove = len(stop_word)
    #                     if output_str[-length_to_remove:] == stop_word:
    #                         output_str = output_str[:-length_to_remove]
    #                 output_str = output_str.strip()
    #             output_str_list.append(output_str)

    #     if self.device:
    #         torch.cuda.empty_cache()

    #     return output_str_list
    
    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def lm_score(self, input_text1, input_text2, 
                start_layer, 
                end_layer,
                attn_alpha=1.5,
                token_enhance=None,
                token_weaken=None,
                pmi=False, 
                max_new_tokens=256, 
                top_p=0.95, 
                top_k=0, 
                temperature=0.8, 
                verbose=True, 
                remove_stop_words=False, 
                relative_top=0.1, 
                relative_top_value=-1000.0, 
                post_softmax=True, 
                alpha=1,
                **kwargs,
                ):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_encoding = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            prefix_encoding = self.tokenizer(input_text1, return_tensors="pt").to(self.device)
            input_ids = input_encoding['input_ids']
            prefix_ids = prefix_encoding['input_ids']
            attention_mask = input_encoding['attention_mask']
            
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            completion_len =len(continue_ids)
                
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[0].squeeze(0)
            outputs = outputs.log_softmax(-1)  
            logits = outputs[prefix_ids.shape[-1] - 1: -1, :]
            
            # ### cd
            query_tokens = prefix_ids.tolist()
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
            
            outputs_modified = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[0].squeeze(0)
            outputs_modified = outputs_modified.log_softmax(-1)  # logits to log probs
            logits_modified = outputs_modified[prefix_ids.shape[-1] - 1: -1, :]

            final_logits = logits-logits_modified
            final_logits = logits + alpha * final_logits 
            
            if alpha == 0:
                final_logits = logits
            else:
                final_logits = final_logits.log_softmax(dim=-1)
        
            log_probs = final_logits[range(logits.shape[0]), continue_ids].sum().item()     
            llama_restore(self.model, 'llama', start_layer=start_layer, end_layer=end_layer)   
            
        return log_probs, completion_len