import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk import pos_tag
import string
import json
import re
from attention_adaptive import llama_modify_adaptive
def parse_word_list(s): 
    return [x.strip() for x in s.split(',') if x.strip()]

def update_entry(target_dict, key, attn):
    if key in target_dict:
        target_dict[key]['attn']+=attn
        target_dict[key]['num']+=1
    else:
        target_dict[key] = {
            'attn': attn.copy(),
            'num': 1
        }
    return target_dict

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
    {'following', 'did', 'do', 'doing','does', 'according', 'which','q','a'}  # 保留原有其他词
)


def process_attn(models,tensor,tokens,token_dict,target_token):
    model_type = models.lower()
    if 'llama3' in model_type:
        start_sym = 'Ġ'
    elif 'llama' in model_type:
        start_sym = '▁'
    elif 'qwen' in model_type:
        start_sym = 'Ġ'
    is_llama3 = ('llama3' or 'qwen') in model_type
    # 辅助函数：标点符号判断
    punctuation = set(string.punctuation)
    def is_punct(tok):
        return all(c in punctuation for c in tok)

    # 步骤 1: 合并子词为完整单词，记录原始索引
    current_word, current_indices = None, []
    words, columns_list = [], []
    
    force_new_word = False
    for i, token in enumerate(tokens):
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

    processed_columns = []
    for cols in columns_list:
        summed = tensor[:, cols].sum(dim=1, keepdim=True)
        processed_columns.append(summed)
        # ave = tensor[:, cols].mean(dim=1, keepdim=True)
        # processed_columns.append(ave)
    
    new_tensor = torch.cat(processed_columns, dim=1)
    # 步骤 2: 词性标注与分类
    #print(words)
    tagged = pos_tag(words)
    punctuation_set, concept_n, concept_adj, no_concept, concept_v, token_dict, target_idx= token_set(tagged,token_dict,target_token)
    for token in target_idx:
        if target_idx[token]:
            target_idx[token]=[num for idx in target_idx[token] for num in columns_list[idx]]
    if punctuation_set:
        ### token idx
        p_idx = [num for idx in punctuation_set for num in columns_list[idx]]
        punctuation_set_t = new_tensor[:, punctuation_set]
        p=punctuation_set_t.sum(dim=1,keepdim=True)
    else:
        p_idx = []
        p=None
        
    if concept_n:
        cn_idx = [num for idx in concept_n for num in columns_list[idx]]
        concept_n_t = new_tensor[:, concept_n]
        cn=concept_n_t.sum(dim=1,keepdim=True)
    else:
        cn_idx = []
        cn=None
        
    if concept_adj:
        ca_idx = [num for idx in concept_adj for num in columns_list[idx]]
        concept_adj_t = new_tensor[:, concept_adj]
        ca=concept_adj_t.sum(dim=1,keepdim=True)
    else:
        ca_idx = []
        ca=None
        
    if no_concept:
        no_concept_t = new_tensor[:, no_concept]
        nc=no_concept_t.sum(dim=1,keepdim=True)
        nc_idx = [num for idx in no_concept for num in columns_list[idx]]
    else:
        nc_idx = []
        nc=None
        
    if concept_v:
        concept_v_t = new_tensor[:, concept_v]
        cv=concept_v_t.sum(dim=1,keepdim=True)
        cv_idx = [num for idx in concept_v for num in columns_list[idx]]
    else:
        cv_idx = []
        cv=None
            
    if 0 not in p_idx:
        p_idx.append(0)
    
    for ind in [cn_idx, ca_idx, cv_idx, nc_idx]:
        if ind is not None and 0 in ind:
            ind[:] = [x for x in ind if x !=0]

    # merged_p_indices = []
    # for i, (word_ori, _) in enumerate(tagged):
    #     word = (r'^[\\\'"\-]+', '', word_ori)
    #     word = word.lower()
    #     #print(word)
    #     if word in ['<|begin_of_text|>','<s>', None, '"', ':', '.', ',', '?', '<0x0a>', '']:
    #         merged_p_indices.extend(columns_list[i])

    # if merged_p_indices:
    #     merged_p = tensor[:,merged_p_indices].sum(dim=1, keepdim=True)
    # else:
    #     merged_p = None
            
    return p, cn, ca, cv, nc, p_idx, cn_idx, ca_idx, cv_idx, nc_idx, token_dict, target_idx

def token_set(tagged, token_dict,target_token):
    punctuation_set = []
    concept_n = []
    concept_adj = []
    concept_v = []
    no_concept = []
    ###
    target_idx={}
    for j in target_token:
        target_idx.update({j: []})

    for i, (word_ori, tag) in enumerate(tagged):
        word = re.sub(r'^[\\\'"\-]+', '', word_ori)
        word = word.lower()
        
        #print(word)
        if word in ['<|begin_of_text|>','<s>', None, '"', ':', '.', ',', '?', '<0x0a>', ''] or word.startswith('_') or word.startswith('Ġ'):
            #print('p')
            punctuation_set.append(i)
            sub_dict=token_dict['p']
            sub_dict[word] = sub_dict.get(word,0)+1
            if word in target_token:
                target_idx[word].append(i)
        elif tag in ['IN', 'DT', 'TO', 'MD', 'CC'] or tag.startswith('W') or word in all_words:
            no_concept.append(i)
            sub_dict=token_dict['n']
            sub_dict[word] = sub_dict.get(word,0)+1
            if word in target_token:
                target_idx[word].append(i)
        elif (tag.startswith('N') and word != 'which') or tag in ['CD', 'PRP', 'PRP$', 'POS']:
            concept_n.append(i)
            sub_dict=token_dict['c']
            sub_dict[word] = sub_dict.get(word,0)+1
            if word in target_token:
                target_idx[word].append(i)
        elif tag.startswith('J'):
            concept_adj.append(i)
            sub_dict=token_dict['c']
            sub_dict[word] = sub_dict.get(word,0)+1
            if word in target_token:
                target_idx[word].append(i)
        elif tag.startswith('V'):
            concept_v.append(i)
            sub_dict=token_dict['c']
            sub_dict[word] = sub_dict.get(word,0)+1
            
            if word in target_token:
                #print(target_idx[word],type(target_idx[word]))
                target_idx[word].append(i)
                #print(target_idx[word],type(target_idx[word]))
    return punctuation_set, concept_n, concept_adj, no_concept, concept_v, token_dict, target_idx

def get_token_indices(models,tokens,token_dict,target_token):
    model_type = models.lower()
    if 'llama3' in model_type:
        start_sym = 'Ġ'
    elif 'llama' in model_type:
        start_sym = '▁'
    elif 'qwen' in model_type:
        start_sym = 'Ġ'
    is_llama3 = ('llama3' or 'qwen') in model_type
    # 辅助函数：标点符号判断
    punctuation = set(string.punctuation)
    def is_punct(tok):
        return all(c in punctuation for c in tok)

    # 步骤 1: 合并子词为完整单词，记录原始索引
    current_word, current_indices = None, []
    words, columns_list = [], []
    
    force_new_word = False
    for i, token in enumerate(tokens):
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
    punctuation_set, concept_n, concept_adj, no_concept, concept_v, token_dict, target_idx= token_set(tagged,token_dict,target_token)
    for token in target_idx:
        if target_idx[token]:
            target_idx[token]=[num for idx in target_idx[token] for num in columns_list[idx]]
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

    return p_idx, cn_idx, ca_idx, cv_idx, nc_idx, token_dict, target_idx

# def lm_score(model_name, model, tokenizer, input_text1, input_text2, demo,question,idx, token_dict,target_token, target_dict,Concept,No_concept,P):
#     input_text = input_text1 + input_text2
#     input_encoding = tokenizer(input_text, return_tensors="pt").to(model.device)
#     prefix_encoding = tokenizer(input_text1, return_tensors="pt").to(model.device)
#     demo_encoding = tokenizer(demo, return_tensors="pt").to(model.device)
#     question_encoding = tokenizer(question, return_tensors="pt").to(model.device)

#     input_ids = input_encoding['input_ids']
#     demo_ids = demo_encoding['input_ids']
#     question_ids = question_encoding['input_ids']
#     prefix_ids = prefix_encoding['input_ids']
#     continue_ids = input_ids[0, prefix_ids.shape[-1]:]
#     attention_mask = input_encoding['attention_mask']
#     layer_attn = None
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True)
#     attentions = outputs.attentions

#     tokens_text = []
#     ###
#     #query_tokens = prefix_ids.tolist()
#     ###
#     query_tokens = question_ids.tolist()
#     demo_tokens = demo_ids.tolist()
#     query_text = tokenizer.convert_ids_to_tokens(query_tokens[0], skip_special_tokens=False)
#     p_idx, cn_idx, ca_idx, cv_idx, nc_idx,token_dict, target_idx = get_token_indices(model_name, query_text,token_dict, target_token)
#     #print(attentions.size(0))
#     for layer_idx in range(len(attentions)):
#         start_index = prefix_ids.shape[-1] - 1
#         attn = attentions[layer_idx].mean(dim=1)[0]
#         attn_first = attn[start_index, :prefix_ids.shape[-1]]
#         if layer_idx == 0:
#             layer_attn=attn_first
#             #print(attn_first.size())
#         else:
#             layer_attn = torch.vstack((layer_attn, attn_first))


#     for target in target_token:
#         if len(target_idx[target])!=0:
#             select_attn=layer_attn[:,target_idx[target]]
#             select_attn=select_attn.mean(dim=1,keepdim=True).cpu().detach().numpy()
#             target_dict=update_entry(target_dict, target, select_attn)

    
#     p, cn, ca, cp, nc, p_idx, cn_idx, ca_idx, cv_idx, nc_idx,token_dict, target_idx = process_attn(model_name, layer_attn,query_text,token_dict, target_token)

#     processed = []
#     for t in [p, cn, ca, cp, nc]:
#         if t == None:
#             zero = torch.zeros(32,1).to(device)
#             processed.append(zero)
#         else:
#             processed.append(t)
    
#     concept = processed[1] + processed[2] + processed[3]
#     if Concept != None:
#         Concept+=concept
#         No_concept += processed[-1]
#         P += processed[0]
#         #Concept_N += processed[1]
#     else:
#         Concept=concept
#         No_concept = processed[-1]
#         P = processed[0]
#         #Concept_N = processed[1]

#     #print(layer_attn.size())
#     ###token
#     if layer_attn is not None:
#         plt.figure(figsize=(14,10))
#         if not os.path.exists(f"/root/zhujingze/Concept/vis/{model_name}"):
#             os.makedirs(f"/root/zhujingze/Concept/vis/{model_name}")
#         num_layers = layer_attn.size(0)
#         layer_labels = [f"{i}" for i in range(num_layers)]
#         token_labels = query_text
#         ori_hm=sns.heatmap(
#             layer_attn.cpu().detach().numpy().T,
#             annot=True,
#             fmt=".1f",
#             xticklabels=layer_labels,
#             yticklabels=token_labels,
#             cmap="coolwarm",
#             vmin=0,
#             vmax=0.01,
#             # vmin=-15,
#             # vmax=5,
#             #cbar_kws={'location': 'top', 'aspect':20},
#         )
#         #ori_hm.xaxis.tick_top()
#         #ori_hm.xaxis.set_label_position('top')
#         #ori_hm.set_title(f"Correct token: {concept_token}, Contrastive Token: {other_token}", fontsize=14, fontweight='bold', pad=5)                 # 调整标题与图的间距
#         ori_hm.set_xlabel("Layer",fontsize=16,fontweight='bold')
#         ori_hm.set_ylabel("Token",fontsize=16,fontweight='bold')
#         ori_hm.tick_params(axis='y', labelsize=14, rotation=0)           # 可选：倾斜角度防止重叠
#         plt.savefig(f"/root/zhujingze/Concept/vis/{model_name}/{idx}.png")
#         plt.close()
#     return token_dict, target_dict,Concept,No_concept,P

##pic2
def lm_score(model_name, model, tokenizer, input_text1, input_text2, demo, question, idx, token_dict, target_token, target_dict, Concept, No_concept, P):
    input_text = input_text1 + input_text2
    input_encoding = tokenizer(input_text, return_tensors="pt").to(model.device)
    #print('1',input_text1,'2',input_text)
    # 获取input_text1的编码和offset_mapping以定位question部分
    prefix_encoding = tokenizer(input_text1, return_tensors="pt", return_offsets_mapping=True).to(model.device)
    demo_encoding = tokenizer(demo, return_tensors="pt").to(model.device)
    question_encoding = tokenizer(question, return_tensors="pt").to(model.device)

    input_ids = input_encoding['input_ids']
    demo_ids = demo_encoding['input_ids']
    question_ids = question_encoding['input_ids']
    prefix_ids = prefix_encoding['input_ids']
    continue_ids = input_ids[0, prefix_ids.shape[-1]:]
    attention_mask = input_encoding['attention_mask']
    layer_attn = None

    # 使用offset_mapping确定question在input_text1中的token索引
    offset_mapping = prefix_encoding.offset_mapping[0].cpu().numpy()
    demo_length_chars = len(demo)  # 获取demo的字符长度
    question_start_char = demo_length_chars  # question在input_text1中的起始字符位置

    question_token_indices = []
    for i, (start, end) in enumerate(offset_mapping):
        # 确定属于question部分的token索引
        if start >= question_start_char:
            question_token_indices.append(i)
        elif start < question_start_char and end > question_start_char:
            # 处理跨边界的token，归入question部分
            question_token_indices.append(i)
    ###<s>
    del_list=[0,1,14]
    question_token_indices=[val for i,val in enumerate(question_token_indices) if i not in del_list]
    question_token_indices.insert(0,0)
    # 截取question部分的token IDs和文本
    # question_in_prefix_ids = prefix_ids[0, question_token_indices]
    # query_tokens = question_in_prefix_ids.tolist()
    # query_text = tokenizer.convert_ids_to_tokens(query_tokens, skip_special_tokens=False)

    query_tokens = prefix_ids.tolist()
    query_text = tokenizer.convert_ids_to_tokens(query_tokens[0], skip_special_tokens=False)
    # 使用question的token处理indices和attention
    p_idx, cn_idx, ca_idx, cv_idx, nc_idx, token_dict, target_idx = get_token_indices(model_name, query_text, token_dict, target_token)
    question_in_prefix_ids = prefix_ids[0, question_token_indices]
    query_tokens = question_in_prefix_ids.tolist()
    query_text = tokenizer.convert_ids_to_tokens(query_tokens, skip_special_tokens=False)
    ###mod
    token_weaken_idx = cn_idx + ca_idx + cv_idx
    ac_idx=token_weaken_idx
    # print(token_weaken_idx)
    llama_modify_adaptive(
        model,
        'llama',
        start_layer=5,
        end_layer=16,
        use_attn=True,
        alpha=0,
        first_token_idx=prefix_ids.shape[-1] - 1,
        token_enhance=None,
        token_weaken=token_weaken_idx,
        n_idx=nc_idx,
        c_idx=ac_idx,
        s_idx=p_idx,
        th=0,
        #ave_token= torch.zeros((1, 253 - prefix_ids.shape[-1]+1), dtype=torch.float32), # C,P,N
        sink = False,
        ema = False,
        wk_c = True,
        special_layers=[0,1,2,3,4]
    )

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True)
    attentions = outputs.attentions
    logits=outputs.logits[0]
    logits=logits.log_softmax(-1)
    #print(logits.size())
    #print(tokenizer(["Yes"]), tokenizer(["No"]),tokenizer(["yes"]),tokenizer(["no"]))
    print(logits[prefix_ids.shape[-1]-1,3869],logits[prefix_ids.shape[-1]-1,4874],logits[prefix_ids.shape[-1]-1,1939],logits[prefix_ids.shape[-1]-1,694])
    # logits_out = logits[len(prefix_ids)-1]
    # print(torch.argmax(logits_out))
    # print(tokenizer.convert_ids_to_tokens([917], skip_special_tokens=False))
    #print(tokenizer("the", return_tensors="pt"),tokenizer("people", return_tensors="pt"))

    # 处理各层的注意力，仅关注question部分的token
    for layer_idx in range(len(attentions)):
        start_index = prefix_ids.shape[-1] - 1  # input_text1的最后一个token位置
        attn = attentions[layer_idx].mean(dim=1)[0]  # 平均多头注意力
        # 提取start_index位置对question部分token的注意力
        attn_first = attn[start_index, question_token_indices]
        #attn_first = attn[start_index, :prefix_ids.shape[-1]]
        if layer_idx == 0:
            layer_attn = attn_first.unsqueeze(0)  # 保持二维结构以便后续拼接
        else:
            layer_attn = torch.vstack((layer_attn, attn_first))
    # 更新target_dict中的注意力值
    for target in target_token:
        if len(target_idx.get(target, [])) != 0:
            select_attn = layer_attn[:, target_idx[target]]
            select_attn = select_attn.mean(dim=1, keepdim=True).cpu().detach().numpy()
            target_dict = update_entry(target_dict, target, select_attn)

    # 处理注意力并更新概念相关变量
    p, cn, ca, cp, nc, p_idx, cn_idx, ca_idx, cv_idx, nc_idx, token_dict, target_idx = process_attn(model_name, layer_attn, query_text, token_dict, target_token)
    # p = p / len(p_idx)
    # cn = cn / len(cn_idx)
    # ca = ca / len(ca)
    # cp= cp / len(cp)
    # nc = nc / len(nc)

    ###merged
    full_prefix_tokens = tokenizer.convert_ids_to_tokens(prefix_ids[0], skip_special_tokens=False)
    full_p_idx, full_cn_idx, full_ca_idx, full_cv_idx, full_nc_idx, _, full_target_idx = get_token_indices(
        model_name, full_prefix_tokens, token_dict, ['<|begin_of_text|>','<s>', None, '"', ':', '.', ',', '?', '<0x0a>', '']
    )

    # 提取merged_p的注意力（基于整个prefix_ids）
    merged_p_attentions = []
    for layer_idx in range(len(attentions)):
        start_index = prefix_ids.shape[-1] - 1  # 整个prefix的最后一个token位置
        attn = attentions[layer_idx].mean(dim=1)[0]  # 平均多头注意力
        # 获取所有target_token对应的索引
        merged_indices = [idx for token in target_token for idx in full_target_idx.get(token, [])]
        merged_attn = attn[start_index, merged_indices].sum()
        merged_p_attentions.append(merged_attn)

    merged_p = torch.tensor(merged_p_attentions).unsqueeze(1)  # 转换为[layers, 1]形状

    # 更新target_dict
    if len(merged_indices) > 0:
        merged_attn_np = merged_p.mean(dim=1, keepdim=True).cpu().detach().numpy()
        target_dict = update_entry(target_dict, 'merged_p', merged_attn_np)

    processed = []
    for t in [p, cn, ca, cp, nc]:
        processed.append(t if t is not None else torch.zeros(layer_attn.size(0), 1).to(model.device))

    concept = processed[1] + processed[2] + processed[3]
    if Concept is not None:
        Concept += concept
        No_concept += processed[-1]
        P += processed[0]
    else:
        Concept = concept
        No_concept = processed[-1]
        P = processed[0]

    # 可视化部分仅显示question的token
    
    if not os.path.exists(f"/root/zhujingze/Concept/vis/{model_name}"):
        os.makedirs(f"/root/zhujingze/Concept/vis/{model_name}")
    if layer_attn is not None:
        # plt.figure(figsize=(10, 10))
        # #plt.figure(figsize=(16, 100))
        # num_layers = layer_attn.size(0)
        # layer_labels = [f"{i}" for i in range(num_layers)]
        # sns.heatmap(
        #     layer_attn.cpu().detach().numpy(),
        #     annot=False,
        #     #fmt=".1f",
        #     xticklabels=query_text,
        #     yticklabels=layer_labels,  # 使用question的token标签
        #     cmap="Blues",
        #     vmin=0,
        #     vmax=0.05,
        # )
        # plt.xlabel("Token", fontsize=16, fontweight='bold')
        # plt.ylabel("Layer", fontsize=16, fontweight='bold')
        # plt.tick_params(axis='y', labelsize=14, rotation=0)
        plt.figure(figsize=(10, 10))
        num_layers = layer_attn.size(0)
        layer_labels = [f"{i+1}" for i in range(num_layers)]  # 显示从1开始

        # 构造 y 轴显示的 index：每5层显示一次
        yticks_index = list(range(0, num_layers, 5))
        yticks_labels = [str(i) for i in yticks_index]

        # 创建子图用于控制 colorbar 位置
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            layer_attn.cpu().detach().numpy(),
            ax=ax,
            annot=False,
            xticklabels=query_text,
            yticklabels=layer_labels,
            cmap="Blues",
            vmin=0,
            vmax=0.1,
            cbar_kws={"location": "left", "pad": 0.08}  # colorbar 放左边
        )

        # 设置坐标轴标签和字体样式
        #ax.set_xlabel("Token", fontsize=18, fontweight='bold')
        ax.set_ylabel("Layer", fontsize=18, fontweight='bold')

        # 设置x轴刻度字体
        ax.tick_params(axis='x', labelsize=14)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')

        # 设置y轴刻度字体、只显示指定层
        ax.set_yticks(yticks_index)
        ax.set_yticklabels(yticks_labels, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)

        plt.tight_layout()
        plt.savefig(f"/root/zhujingze/Concept/vis/{model_name}/{idx}.png")
        plt.close()

    return token_dict, target_dict, Concept, No_concept, P


from utils_truthfulqa import *
import transformers
from tqdm import tqdm, trange
import argparse
import json
import numpy as np
import warnings
transformers.logging.set_verbosity(40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data_path", type=str, default="Data/TruthfulQA")
    parser.add_argument("--target_token", type=parse_word_list,default=[])

    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    target_token = args.target_token
    #print(target_token)
    #target_token=parse_word_list(target_token)
    model_name_input = os.path.basename(model_name.rstrip('/'))
    token_dict = {
        'p':{},
        'c':{},
        'n':{}
    }
    target_dict = {}
    Concept = None
    Concept_N = None
    No_concept = None
    P = None
    # Get test file
    '''
    The StrategyQA dataset includes the followings files:
        strategyqa_train.json: The training set of StrategyQA, which includes 2,290 examples.
        strategyqa_train_paragraphs.json: Paragraphs from our corpus that were matched as evidence for examples in the training set.
        strategyqa_train_filtered.json: 2,821 additional questions, excluded from the official training set, that were filtered by our solvers during data collection (see more details in the paper).
        strategyqa_test.json: The test set of StrategyQA, which includes 490 examples.
    Here we only need the test set.
    '''
    list_data_dict = load_truthfulqa_dataset(args.data_path)
    #kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
    # num_gpus = int(num_gpus)
    # if num_gpus != 1:
    #     kwargs.update({
    #         "device_map": "auto",
    #         "max_memory": {i: f"{args.max_gpu_memory}GiB" for i in range(num_gpus)},
    #     })
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with torch.no_grad():
        for idx, sample in tqdm(enumerate(list_data_dict[662:663])):
            ref_true = []
            ref_t = split_multi_answer(sample['answer_true'])
            ref_true.append(ref_t)
            for temp_ans in ref_true:
                prompt, answer,demo,question = build_prompt_and_answer(sample['question'], temp_ans)
                token_dict,target_dict,Concept,No_concept,P = lm_score(model_name_input, model, tokenizer, prompt, answer,demo,question, idx, token_dict, target_token, target_dict,Concept,No_concept,P)
    sorted_dict = {
        category: dict(sorted(sub_dict.items(), key=lambda x: x[1], reverse=True))
        for category, sub_dict in token_dict.items()
    }
    with open(f"/root/zhujingze/Concept/vis/{model_name_input}.json", "w", encoding="utf-8") as f:
        json.dump(sorted_dict,f,indent=4)
    
    # # 获取所有token和层数
    # tokens = list(target_dict.keys())
    # #tokens.insert(0,'<s>')
    # num_layers = len(model.model.layers) #len(next(iter(target_dict.values()))['attn'])  # 自动获取层数

    # # 创建存储平均注意力的矩阵 (layers × tokens)
    # heatmap_data = np.zeros((num_layers, len(tokens)))

    # # 填充数据矩阵
    # tokens=['a','q','the','an','per','any','is','was','been','am','being','would','might','could','your','their','its','of','by','in','with','over','at','through','out','.','?','<0x0a>','<s>','merged_p']
    # #tokens=['<s>']
    # for col_idx, token in enumerate(tokens):
    #     entry = target_dict[token]
    #     avg_attn = entry['attn'] / entry['num']
    #     heatmap_data[:, col_idx] = avg_attn.flatten()

    # # 步骤2：绘制热力图
    # # --------------------------------------------------
    # plt.figure(figsize=(25,10))

    # # 生成热力图
    # ax = sns.heatmap(
    #     heatmap_data,
    #     annot=True,          # 显示数值
    #     fmt=".2f",          # 数值格式
    #     cmap="YlGnBu",      # 颜色方案
    #     vmin=0,
    #     vmax=0.05,
    #     xticklabels=tokens, # x轴标签
    #     yticklabels=[f"{i}" for i in range(num_layers)]  # y轴标签
    # )

    # # 调整坐标轴方向（可选）
    # #ax.invert_yaxis()  # 使Layer 1显示在最上方
    # ax.set_xlabel("Tokens",fontsize=16,fontweight='bold')
    # ax.set_ylabel("Layers",fontsize=16,fontweight='bold')
    # ax.tick_params(axis='x', labelsize=10, rotation=0)  
    # # 添加标签和标题
    # plt.title("Average Attention Heatmap")

    # plt.tight_layout()
    # plt.savefig(f"/root/zhujingze/Concept/vis/{model_name_input}.png")
    # plt.close()
    # with open(f"/root/zhujingze/Concept/vis/{model_name_input}_target.json", "w", encoding="utf-8") as f:
    #     json.dump(target_dict,f,indent=4)


    layers = np.arange(len(model.model.layers))
    concept_ratios_single = [Concept[lay_idx].item() for lay_idx in layers]
    p_ratios_single = [P[lay_idx].item() for lay_idx in layers]
    nc_ratios_single = [No_concept[lay_idx].item() for lay_idx in layers]
    #concept_n_ratios_single = [processed[1][lay_idx].item() for lay_idx in layers]
    #print(concept_ratios_single)
    # 创建画布
    plt.figure(figsize=(10, 6))
    concept_ratios_single = [val/(idx+1) for val in concept_ratios_single]
    p_ratios_single = [val/(idx+1) for val in p_ratios_single]
    nc_ratios_single = [val/(idx+1) for val in nc_ratios_single]
    # 绘制三条折线
    plt.plot(layers, concept_ratios_single, marker='o', label='Concept Ratio')
    plt.plot(layers, p_ratios_single, marker='s', label='P Ratio')
    plt.plot(layers, nc_ratios_single, marker='^', label='No Concept Ratio')
    plt.axhline(y=0.1,color='gray',linestyle='--',linewidth=1,alpha=0.5)
    plt.axhline(y=0.05,color='gray',linestyle='--',linewidth=1,alpha=0.5)
    #print('c',concept_ratios_single,'p',p_ratios_single,'n',nc_ratios_single)
    #plt.plot(layers, concept_n_ratios_single, marker='x', label='Concept_N Ratio', alpha=0.5)
        
    # 添加标签和标题
    plt.xlabel('Layer Index')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"/root/zhujingze/Concept/vis/ratio.png", dpi=300, bbox_inches='tight')
    plt.close()
