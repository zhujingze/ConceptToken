import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from attention import llama_modify, llama_restore
from dataset_mmlu import MultipleChoiceDataset
from dataset_mmlu_wo_option import MultipleChoiceConcatWODataset
from dataset_mmlu_lens import MultipleChoiceLens
from torch.utils.data import DataLoader
from matplotlib import gridspec
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
    {'following', 'did', 'do', 'does', 'according'}  # 保留原有其他词
)

def set_encoder(obj):
    """自定义序列化处理器：将集合转为列表"""
    if isinstance(obj, set):
        return list(obj)  # 集合 → 列表
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def logits_change_nm(layer_prob_word):
    first_max_word = layer_prob_word.max()
    first_min_word = layer_prob_word.min()
    layer_prob_word = (layer_prob_word-first_min_word) / (first_max_word-first_min_word)
    return layer_prob_word

def process_diff(token_seq, logits_source, answer_idx, other_idx, b_idx, true_ls, false_ls):
    # 获取最小长度和差异位置
    min_len = min(len(token_seq[answer_idx]), len(token_seq[other_idx]))
    find_diff = False
    
    # 查找第一个差异位置
    for token_diff_idx in range(min_len):
        if token_seq[answer_idx][token_diff_idx] != token_seq[other_idx][token_diff_idx]:
            break
    else:
        token_diff_idx = min_len - 1  # 没有找到差异时使用最后一个位置
    
    # 处理logits
    process_logits_norm = lambda x: logits_change_nm(x)
    process_logits = lambda x: x

    # 获取对应logits
    logits_true = process_logits(logits_source[answer_idx][:, token_diff_idx])
    logits_false = process_logits(logits_source[other_idx][:, token_diff_idx])

    logits_true_norm = process_logits_norm(logits_source[answer_idx][:, token_diff_idx])
    logits_false_norm = process_logits_norm(logits_source[other_idx][:, token_diff_idx])
    # 处理不同b_idx的情况
    #if b_idx == 0:
    #print('t',true_ls)
    #print(type(true_ls))
    if isinstance(true_ls, np.ndarray):
        return true_ls+logits_true_norm, false_ls+logits_false_norm, logits_true, logits_false, token_diff_idx
    else:
        return logits_true_norm, logits_false_norm, logits_true, logits_false, token_diff_idx
    
def token_set(tagged, token_word):
    punctuation_set = []
    concept_n = []
    concept_adj = []
    concept_v = []
    no_concept = []
    ###
    
    for i, (word_ori, tag) in enumerate(tagged):
        word = re.sub(r'^[\\\'"\-]+', '', word_ori)
        if word in ['<s>', None, '"', ':', '.', ',', '?', '<0x0A>', ''] or word.startswith('_'):
            punctuation_set.append(i)
            token_word['pword'].add(word)
        elif tag in ['IN', 'DT', 'TO', 'MD', 'CC'] or tag.startswith('W') or word.lower() in all_words:
            no_concept.append(i)
            token_word['ncword'].add(word)
        elif (tag.startswith('N') and word != 'Which') or tag in ['CD', 'PRP', 'PRP$', 'POS']:
            concept_n.append(i)
            token_word['cnword'].add(word)
        elif tag.startswith('J'):
            concept_adj.append(i)
            token_word['caword'].add(word)
        elif tag.startswith('V'):
            concept_v.append(i)
            token_word['cvword'].add(word)
    return punctuation_set, concept_n, concept_adj, no_concept, concept_v
### attn操作前定义token
def get_token_indices(tokens):
    """ 根据 tokens 生成词性分类对应的原始 token 索引 """
    # 辅助函数：标点符号判断
    punctuation = set(string.punctuation)
    def is_punct(tok):
        return all(c in punctuation for c in tok)

    # 步骤 1: 合并子词为完整单词，记录原始索引
    current_word, current_indices = None, []
    words, columns_list = [], []
    
    for i, token in enumerate(tokens):
        if token == '<0x0A>':  # 换行符特殊处理
            if current_word:
                words.append(current_word)
                columns_list.append(current_indices)
                current_word, current_indices = None, []
            continue
            
        if token.startswith('▁') or token == '<s>' or is_punct(token):
            if current_word:  # 保存上一个词
                words.append(current_word)
                columns_list.append(current_indices)
            current_word = token.lstrip('▁')  # 去除子词标记
            current_indices = [i]
        else:
            current_word = (current_word or '') + token
            current_indices.append(i)
            
    if current_word:  # 处理最后一个词
        words.append(current_word)
        columns_list.append(current_indices)

    if not words:
        return None

    # 步骤 2: 词性标注与分类
    tagged = pos_tag(words)
    
    # 简化版分类逻辑（需根据实际 token_set 实现调整）
    p_indices, cn_indices, ca_indices, cv_indices, nc_indices = [], [], [], [], []
    for idx, (word, tag) in enumerate(tagged):
        word_clean = re.sub(r'^[\\\'"\-]+', '', word)
        if word_clean in {'<s>', '<0x0A>'} or is_punct(word):
            p_indices.extend(columns_list[idx])
        elif tag.startswith('N') or tag in ['CD', 'PRP']:
            cn_indices.extend(columns_list[idx])
        elif tag.startswith('J'):
            ca_indices.extend(columns_list[idx])
        elif tag.startswith('V'):
            cv_indices.extend(columns_list[idx])
        else:
            nc_indices.extend(columns_list[idx])

    return p_indices, cn_indices, ca_indices, cv_indices, nc_indices

def process_concept(tokens):
    punctuation = set(string.punctuation)

    def is_punctuation(tok):
        return all(c in punctuation for c in tok)

    # 组合tokens成单词并记录对应列索引
    current_word = None
    current_columns = []
    words = []
    columns_list = []

    for i, token in enumerate(tokens):
        if is_punctuation(token):
            continue  # 跳过标点符号

        if token.startswith('▁'):
            # 新单词开始
            if current_word is not None:
                words.append(current_word)
                columns_list.append(current_columns)
            current_word = token.lstrip('▁')  # 去掉前缀
            current_columns = [i]  # 记录当前token的索引
        else:
            # 继续当前单词
            if current_word is not None:
                current_word += token
                current_columns.append(i)
            else:
                # 处理没有前缀的特殊情况（如首个token）
                current_word = token
                current_columns = [i]

    # 添加最后一个单词
    if current_word is not None:
        words.append(current_word)
        columns_list.append(current_columns)

    if not words:
        return []

    # 词性标注并筛选名词
    # tagged = pos_tag(words)
    # noun_indices = [i for i, (_, tag) in enumerate(tagged) if tag.startswith('N')]

    # 提取每个名词对应的第一个token的索引
    word_idx = []
    for i in range(len(words)):
        word_idx.append(i)
    first_token_indices = [columns_list[i][0] for i in word_idx]

    return first_token_indices

def process_attn(tensor, tokens, token_word):
    # 处理标点符号判断
    punctuation = set(string.punctuation)
    def is_punctuation(tok):
        return all(c in punctuation for c in tok)
    
    # 第一步：组合tokens成单词并记录对应列索引
    current_word = None
    current_columns = []
    words = []
    columns_list = []
    
    for i, token in enumerate(tokens):
        if token == '<0x0A>':
            if current_word is not None:
                words.append(current_word)
                columns_list.append(current_columns)
                current_word = None
                current_columns = []
            continue
        
        if token.startswith('▁'):
            if current_word is not None:
                words.append(current_word)
                columns_list.append(current_columns)
            current_word = token.lstrip('▁')
            current_columns = [i]
        elif token == '<s>':
            if current_word is not None:
                words.append(current_word)
                columns_list.append(current_columns)
            current_word = token.lstrip('▁')
            current_columns = [i]
        elif is_punctuation(token):
            if current_word is not None:
                words.append(current_word)
                columns_list.append(current_columns)
            current_word = token
            current_columns = [i]
        else:
            if current_word is not None:
                current_word += token
                current_columns.append(i)
            else:
                current_word = token
                current_columns = [i]
    
    # 添加最后一个单词
    if current_word is not None:
        words.append(current_word)
        columns_list.append(current_columns)
    
    if not words:
        return None
    
    # 第二步：处理tensor生成新矩阵
    processed_columns = []
    for cols in columns_list:
        summed = tensor[:, cols].sum(dim=1, keepdim=True)
        processed_columns.append(summed)
        # ave = tensor[:, cols].mean(dim=1, keepdim=True)
        # processed_columns.append(ave)
    
    new_tensor = torch.cat(processed_columns, dim=1)

    # # 第三步：词性标注并筛选名词
    tagged = pos_tag(words)

    punctuation_set, concept_n, concept_adj, no_concept, concept_v = token_set(tagged, token_word)
    
    # 对应token标签汇聚 便于整体操作
    if punctuation_set:
        ### token attn sum
        punctuation_set_t = new_tensor[:, punctuation_set]
        p=punctuation_set_t.sum(dim=1,keepdim=True)
        ### token idx
        p_idx = [num for idx in punctuation_set for num in columns_list[idx]]
    else:
        p=None
        p_idx = None
        
    if concept_n:
        concept_n_t = new_tensor[:, concept_n]
        cn=concept_n_t.sum(dim=1,keepdim=True)
        cn_idx = [num for idx in concept_n for num in columns_list[idx]]
    else:
        cn=None
        cn_idx =None
        
    if concept_adj:
        concept_adj_t = new_tensor[:, concept_adj]
        ca=concept_adj_t.sum(dim=1,keepdim=True)
        ca_idx = [num for idx in concept_adj for num in columns_list[idx]]
    else:
        ca=None
        ca_idx = None
        
    if no_concept:
        no_concept_t = new_tensor[:, no_concept]
        nc=no_concept_t.sum(dim=1,keepdim=True)
        nc_idx = [num for idx in no_concept for num in columns_list[idx]]
    else:
        nc=None
        nc_idx = None
        
    if concept_v:
        concept_v_t = new_tensor[:, concept_v]
        cv=concept_v_t.sum(dim=1,keepdim=True)
        cv_idx = [num for idx in concept_v for num in columns_list[idx]]
    else:
        cv=None
        cv_idx = None
    # noun_indices = [i for i, (_, tag) in enumerate(tagged) if tag.startswith('N')]
    
    # if not noun_indices:
    #     return None
    
    word_idx = []
    for i in range(len(words)):
        word_idx.append(i)
    # 构造最终结果
    filtered_tensor = new_tensor[:, word_idx]
    filtered_words = [words[i] for i in word_idx]
    return filtered_tensor, filtered_words, p, cn, ca, cv, nc, p_idx, cn_idx, ca_idx, cv_idx, nc_idx 

def process_logits(tensor, tokens):
    # 处理标点符号判断
    punctuation = set(string.punctuation)
    
    def is_punctuation(tok):
        return all(c in punctuation for c in tok)
    
    # 第一步：组合tokens成单词并记录对应列索引
    current_word = None
    current_columns = []
    words = []
    columns_list = []
    
    for i, token in enumerate(tokens):
        if token == '<0x0A>':
            if current_word is not None:
                words.append(current_word)
                columns_list.append(current_columns)
                current_word = None
                current_columns = []
            continue

        if is_punctuation(token):
            continue
        
        if token.startswith('▁'):
            if current_word is not None:
                words.append(current_word)
                columns_list.append(current_columns)
            current_word = token.lstrip('▁')
            current_columns = [i]
        else:
            if current_word is not None:
                current_word += token
                current_columns.append(i)
            else:
                current_word = token
                current_columns = [i]
    
    # 添加最后一个单词
    if current_word is not None:
        words.append(current_word)
        columns_list.append(current_columns)
    
    if not words:
        return None
    
    # 第二步：处理tensor生成新矩阵
    processed_columns = []
    for cols in columns_list:
        # summed = tensor[:, cols].sum(dim=1, keepdim=True)
        # processed_columns.append(summed)
        ave = tensor[:, cols].mean(dim=1, keepdim=True)
        processed_columns.append(ave)
    
    new_tensor = torch.cat(processed_columns, dim=1)

    word_idx = []
    for i in range(len(words)):
        word_idx.append(i)
    
    # 构造最终结果
    filtered_tensor = new_tensor[:, word_idx]
    filtered_words = [words[i] for i in word_idx]
    
    return filtered_tensor, filtered_words

def compute_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def compute_accuracy_concat(logits, labels):
    _, predicted = torch.max(logits, 0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def js_divergence(p, q, epsilon=1e-8):

    m = 0.5 * (p + q)  # 中间分布
    kl_p_m = F.kl_div(torch.log(p + epsilon), m, reduction='none').sum(dim=-1)  # KL(P || M)
    kl_q_m = F.kl_div(torch.log(q + epsilon), m, reduction='none').sum(dim=-1)  # KL(Q || M)
    js = 0.5 * (kl_p_m + kl_q_m)  # JS(P || Q)
    return js

def logits_to_onehot(logits):

    max_indices = torch.argmax(logits, dim=-1)  # 找到最大值的位置
    onehot = torch.zeros_like(logits)
    onehot.scatter_(1, max_indices.unsqueeze(1), 1)  # 将最大值位置设为1
    return onehot

def logits_to_onehot_concat(logits):

    max_indices = torch.argmax(logits, dim=-1)  # 找到最大值的位置
    onehot = torch.zeros_like(logits)
    onehot=onehot.unsqueeze(0)
    onehot.scatter_(1, max_indices.unsqueeze(0).unsqueeze(1), 1)  # 将最大值位置设为1
    return onehot

def main(args):
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager",torch_dtype=torch.float16).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
  
    test_filename = args.subject + '_test.csv'
    val_filename = args.subject + '_val.csv'
    
    test_file = os.path.join(os.path.join(args.data_folder, 'test'), test_filename)
    val_file = os.path.join(os.path.join(args.data_folder, 'val'), val_filename)

    if args.method == 'letter':
        dataset = MultipleChoiceDataset(args.subject, test_file, val_file, tokenizer)
    if args.method == 'wo_option':
        dataset = MultipleChoiceConcatWODataset(args.subject, test_file, val_file, tokenizer)
    if args.method in ['lens_ans', 'lens_opt']:
        dataset = MultipleChoiceLens(args.subject, test_file, val_file, tokenizer)
        
    #torch.manual_seed(10)
    train_loader = DataLoader(dataset, batch_size = args.bs, shuffle = False)

    #ori
    start_idx = 1
    total_correct = []
    total_correct_norm = []
    total_scale = []
    total_max = []
    total_entropy = []  # 用于累积每层的熵总和
    total_kl = []       # 用于累积每层的KL散度总和（层i对应与层i-1的KL）
    total_samples = 0
    total_js_matrix = None
    total_consistency_matrix = None
    total_layer_tend = None

    if args.method in ['concat', 'wo_option']:
        for b_idx, batch in enumerate(train_loader):
            # if b_idx >=30:
            #     break
            input = batch["input"]
            prefix = batch["prefix"]
            label = batch["label"].to(device)
            print(f'Q{b_idx} Answer is {chr(65 + int(label.item()))}')
            prefix_encoding = tokenizer(prefix, return_tensors='pt').to(device)
            prefix_id = prefix_encoding["input_ids"]
            # batch
            layer_option = None
            layer_option_norm = None
            for i in range(len(input)):
                input_text = input[i]
                input_encoding = tokenizer(input_text, return_tensors='pt').to(device)
                attention_mask = input_encoding["attention_mask"]
                input_id = input_encoding["input_ids"]
                # 一个batch里答案开始到填充结束
                answer_ids = input_id[:,prefix_id.shape[-1]:]
                #outputs = model(input_ids=input_ids_sample, attention_mask=attention_mask_sample)
                ### one choice all layers
                with torch.no_grad():
                    outputs = model(input_ids=input_id, attention_mask=attention_mask, output_hidden_states=True, num=args.num)

                layer_prob = None
                layer_prob_norm = None
                layers_onehot = []

                for layer_idx in range(outputs.logits.size(0)):
                    num = len(answer_ids[0])
                    # print('num',num)
                    start_index = prefix_id.shape[-1] - 1
                    logits_selected = outputs.logits[layer_idx][start_index:start_index + num]
                    logits_selected = logits_selected.log_softmax(dim=-1)
                    # max_token = torch.argmax(logits_selected, dim=-1)
                    logits_selected = logits_selected[torch.arange(num), answer_ids[0]]
                    #print('logits_selected',answer_ids)
                    logits_selected = logits_selected.sum()
                    logits_selected_norm = logits_selected / num

                    if layer_idx == 0:
                        layer_prob=logits_selected
                        layer_prob_norm=logits_selected_norm
                    else:
                        layer_prob = torch.vstack((layer_prob, logits_selected))
                        layer_prob_norm= torch.vstack((layer_prob_norm, logits_selected_norm))
                if i == 0:
                    layer_option = layer_prob
                    layer_option_norm = layer_prob_norm
                else:
                    layer_option = torch.cat([layer_option, layer_prob], dim=1)
                    layer_option_norm = torch.cat([layer_option_norm, layer_prob_norm], dim=1)

            for idx, row in enumerate(layer_option_norm):

                max_val = torch.max(row)
                max_val = max_val.item()

                if len(total_max) < (idx + 1):
                    total_max.append(max_val)
                else:
                    total_max[layer_idx] += max_val

                if idx >= start_idx - 1:
                    onehot = logits_to_onehot_concat(row)
                    layers_onehot.append(onehot.detach())  # 避免梯度计算

                batch_accuracy_norm = compute_accuracy_concat(row, label)
                #_, predicted_norm = torch.max(row, 0)
                #print(f'Layer {idx + 1} Choose {chr(65 + int(predicted_norm.item()))}')

                if len(total_correct_norm) < (idx + 1):
                        total_correct_norm.append(batch_accuracy_norm * label.size(0))
                else:
                    total_correct_norm[idx] += batch_accuracy_norm * label.size(0)

            for idx, row in enumerate(layer_option):

                batch_accuracy = compute_accuracy_concat(row, label)
                _, predicted = torch.max(row, 0)
                #print(f'Q{b_idx} Layer {idx + 1} Choose {chr(65 + int(predicted.item()))}')

                if len(total_correct) < (idx + 1):
                        total_correct.append(batch_accuracy * label.size(0))
                else:
                    total_correct[idx] += batch_accuracy * label.size(0)

            if layers_onehot:
                layers_onehot = torch.stack(layers_onehot, dim=0)  # (num_layers, batch_size, 4)
                num_layers = layers_onehot.size(0)

                # 初始化一致性矩阵
                consistency_matrix = torch.zeros((num_layers, num_layers), device=device)

                # 计算所有层对之间的一致性
                for i in range(num_layers):
                    for j in range(num_layers):
                        # 计算一致性（相同位置为1的比例）
                        agreement = (layers_onehot[i] == layers_onehot[j]).all(dim=-1).float().mean()
                        consistency_matrix[i, j] = agreement

                # 累积到总矩阵
                if total_consistency_matrix is None:
                    total_consistency_matrix = consistency_matrix.cpu()
                else:
                    total_consistency_matrix += consistency_matrix.cpu()
                if total_layer_tend is None:
                    total_layer_tend = layers_onehot.sum(dim=1).cpu()
                else:
                    total_layer_tend += layers_onehot.sum(dim=1).cpu()

            total_samples += label.size(0)

    for idx, acc in enumerate(total_correct):
        print(f"Layer {idx+1} Acc: {acc / total_samples * 100:.2f}")

    
    #     print("No layers beyond 16 were processed for selection frequency calculation.")
    if args.method in ['lens_ans', 'lens_opt']:
        cd_logits_change = None
        cd_logits_change_r = None
        cd_score_change = None
        cd_score_change_r = None
        cd_logits_change_false = None
        cd_logits_change_r_false = None
        cd_score_change_false = None
        cd_score_change_r_false = None
        
        logits_true = None
        logits_true_modified =None
        logits_false = None
        logits_false_modified = None
        logits_true_word = None
        logits_false_word = None
        total_num = None
        
        layer_acc = [0 for _ in range(len(model.model.layers))]
        layer_acc_modified = [0 for _ in range(len(model.model.layers))]
        
        ###分对错统计
        num_true = 0
        num_false = 0
        ###Token类别
        Concept = None
        Concept_N = None
        No_concept = None
        P = None
        Conceptf = None
        No_conceptf = None
        Pf = None
        token_word = {}
        
        for t_type in ['pword', 'cnword', 'caword', 'caword', 'cvword', 'ncword']:
            token_word[t_type] = set()

        for b_idx, batch in enumerate(train_loader):
            # if b_idx >= 4:
            #     break
            total_num = b_idx + 1
            
            input_text = batch["input"]
            prefix = batch["prefix"]
            option = batch["option"]
            answer_letter = batch["label"]

            prefix_encoding = tokenizer(prefix, return_tensors='pt').to(device)
            prefix_id = prefix_encoding["input_ids"]

            # batch
            layer_option = None
            layer_option_norm = None

            if args.method == 'lens_opt':
                if args.logits_change == True:
                    if args.attn_manipulation == True:
                        llama_restore(model, start_layer=args.start_layer, end_layer=args.end_layer)
                    gs = gridspec.GridSpec(2, 1, height_ratios=[8, 2])  # 比例可自行调整
                    fig = plt.figure(figsize=(12, 14))

                    # 创建子图
                    ax0 = fig.add_subplot(gs[0])  # 第一个子图
                    ax1 = fig.add_subplot(gs[1])  # 第二个子图

                    last_layer_prob = [[] for _ in range(len(model.model.layers))]
                    last_layer_prob_modified = [[] for _ in range(len(model.model.layers))]
                    logits_tmp = []
                    logits_tmp_modified = []
                    logits_tmp_word = []
                    logits_tmp_word_modified = []
                    token_first = []
                    token_first_modified = []
                    token_first_word = []
                    token_first_word_modified = []
                    token_first_sample = None
                    token_first_sample_false = None
                    token_first_word_sample = None
                    token_first_word_sample_false = None
                    for i in range(len(option)):
                        input_text = option[i]
                        input_encoding = tokenizer(input_text, return_tensors='pt').to(device)
                        attention_mask = input_encoding["attention_mask"]
                        input_id = input_encoding["input_ids"]

                        input_id_text = input_id.tolist()
                        #print('input',input_encoding,'text', tokenizer.convert_ids_to_tokens(input_id_text[0], skip_special_tokens=False))
                        # 一个batch里答案开始到填充结束
                        answer_ids = input_id[:,prefix_id.shape[-1]:]
                        #outputs = model(input_ids=input_ids_sample, attention_mask=attention_mask_sample)
                        ### one choice all layers
                        with torch.no_grad():
                            outputs = model(input_ids=input_id, attention_mask=attention_mask, output_hidden_states=True,num=args.num)

                        layer_prob_word = None
                        tokens_text = []
                        tokens = answer_ids.tolist()
                        tokens_text = tokenizer.convert_ids_to_tokens(tokens[0], skip_special_tokens=True)
                        token_first.append(tokens_text)
                        for layer_idx in range(outputs.logits.size(0)):
                            num = len(answer_ids[0])
                            start_index = prefix_id.shape[-1] - 1
                            logits_selected = outputs.logits[layer_idx][start_index:start_index + num]
                            
                            ###判断原最后输出
                            #if layer_idx == 31:
                            logits_selected_prob = logits_selected.log_softmax(dim=-1)

                            logits_selected = logits_selected[torch.arange(num), answer_ids[0]]
                            #print('solo',logits_selected.size())
                            #if layer_idx == 31:
                            logits_selected_prob = logits_selected_prob[torch.arange(num), answer_ids[0]]
                            logits_selected_prob_t = logits_selected_prob.sum()
                            last_layer_prob[layer_idx].append(logits_selected_prob_t)

                            if layer_idx == 0:
                                layer_prob_word = logits_selected
                            else:
                                layer_prob_word = torch.vstack((layer_prob_word, logits_selected))
                            #print('add',layer_prob_word.size())


                        ###某一个sample的所有layer结果
                        filtered_tensor, filtered_words = process_logits(layer_prob_word, tokens_text)
                        layer_prob_word_f = filtered_tensor
                        token_first_word.append(filtered_words)
                        layer_prob_word_f = layer_prob_word_f.cpu().numpy()
                        logits_tmp_word.append(layer_prob_word_f)

                        layer_prob_first = layer_prob_word.cpu().numpy()
                        logits_tmp.append(layer_prob_first)

                    for layer_idx in range(outputs.logits.size(0)):
                        last_layer_prob_each = [tensor.flatten()[0].item() for tensor in last_layer_prob[layer_idx]]
                        last_layer_prob_soerted_each = sorted(last_layer_prob_each, reverse=True)
                        max1_each = last_layer_prob_soerted_each[0]
                        max1_idx_each = last_layer_prob_each.index(max1_each)
                        if answer_letter == max1_idx_each:
                            layer_acc[layer_idx] += 1
                            
                    ###原模型最后层选择的前两个
                    last_layer_prob_ori=last_layer_prob.copy()
                    ori_score  = [sublist[answer_letter] for sublist in last_layer_prob_ori]
                    last_layer_prob = [tensor.flatten()[0].item() for tensor in last_layer_prob[-1]]
                    last_layer_prob_soerted = sorted(last_layer_prob, reverse=True)
                    max1 = last_layer_prob_soerted[0]
                    max2 = last_layer_prob_soerted[1]
                    max1_idx = last_layer_prob.index(max1)
                    max2_idx = last_layer_prob.index(max2)
                    
                    if answer_letter in [max1_idx, max2_idx]:
                        other_idx = max2_idx if answer_letter == max1_idx else max1_idx
                    else:
                        other_idx = max1_idx
                    logits_true, logits_false, token_first_sample, token_first_sample_false, diff_index = process_diff(token_first, logits_tmp, answer_letter, other_idx, b_idx, logits_true, logits_false)
                    other_token = token_first[other_idx][diff_index]
                    ori_score_false  = [sublist[other_idx] for sublist in last_layer_prob_ori]
              
                    if args.attn_manipulation == True:
                        ### 先确定token
                        input_text = option[answer_letter]
                        input_encoding = tokenizer(input_text, return_tensors='pt').to(device)
                        attention_mask = input_encoding["attention_mask"]
                        input_id = input_encoding["input_ids"]
                        answer_ids = input_id[:,prefix_id.shape[-1]:]

                        tokens_text = []
                        tokens = answer_ids.tolist()
                        query_tokens = prefix_id.tolist()

                        tokens_text = tokenizer.convert_ids_to_tokens(tokens[0], skip_special_tokens=True)
                        query_text2 = tokenizer.convert_ids_to_tokens(query_tokens[0], skip_special_tokens=False)
                        concept_token = tokens_text[diff_index]
                        query_text_tmp = query_text2.copy()
                        if diff_index != 0:
                            for ans_idx in range(diff_index):
                                query_text_tmp.append(tokens_text[ans_idx])

                        p_idx, cn_idx, ca_idx, cv_idx, nc_idx = get_token_indices(query_text2)
                        if args.token_enhance == 'cn':
                            token_enhance_idx = cn_idx
                        elif args.token_enhance == 'ac':
                            token_enhance_idx = cn_idx + ca_idx + cv_idx
                        elif args.token_enhance == 'nc':
                            token_enhance_idx = nc_idx
                        else:
                            token_enhance_idx = None
                            
                        if args.token_weaken == 'cn':
                            token_weaken_idx = cn_idx
                        elif args.token_weaken == 'ac':
                            token_weaken_idx = cn_idx + ca_idx + cv_idx
                        elif args.token_weaken == 'nc':
                            token_weaken_idx = nc_idx
                        elif args.token_weaken == 'p':
                            token_weaken_idx = p_idx
                        else:
                            token_weaken_idx = None
                            
                        llama_modify(
                            model,
                            start_layer=args.start_layer,
                            end_layer=args.end_layer,
                            use_attn=True,
                            alpha=args.alpha,
                            #first_token_idx=prefix_id.shape[-1] - 1 + diff_index,
                            first_token_idx=prefix_id.shape[-1] - 1,
                            token_enhance=token_enhance_idx,
                            token_weaken=token_weaken_idx
                        )
                        
                        # 直接处理每个选项的编码和模型推理
                        for i in range(len(option)):
                            input_text = option[i]
                            input_encoding = tokenizer(input_text, return_tensors='pt').to(device)
                            attention_mask = input_encoding["attention_mask"]
                            input_id = input_encoding["input_ids"]

                            input_id_text = input_id.tolist()
                            #print('input',input_encoding,'text', tokenizer.convert_ids_to_tokens(input_id_text[0], skip_special_tokens=False))
                            # 一个batch里答案开始到填充结束
                            answer_ids = input_id[:,prefix_id.shape[-1]:]
                            #outputs = model(input_ids=input_ids_sample, attention_mask=attention_mask_sample)
                            ### one choice all layers
                            with torch.no_grad():
                                outputs = model(input_ids=input_id, attention_mask=attention_mask, output_hidden_states=True,num=args.num)
                            layer_prob_word = None
                            tokens_text = []
                            tokens = answer_ids.tolist()
                            tokens_text = tokenizer.convert_ids_to_tokens(tokens[0], skip_special_tokens=True)
                            token_first_modified.append(tokens_text)
                            for layer_idx in range(outputs.logits.size(0)):
                                num = len(answer_ids[0])
                                start_index = prefix_id.shape[-1] - 1
                                logits_selected = outputs.logits[layer_idx][start_index:start_index + num]
                                
                                ###判断原最后输出
                                #if layer_idx == 31:
                                logits_selected_prob = logits_selected.log_softmax(dim=-1)

                                logits_selected = logits_selected[torch.arange(num), answer_ids[0]]
                                #print('solo',logits_selected.size())
                                #if layer_idx == 31:
                                logits_selected_prob = logits_selected_prob[torch.arange(num), answer_ids[0]]
                                logits_selected_prob_t = logits_selected_prob.sum()
                                last_layer_prob_modified[layer_idx].append(logits_selected_prob_t)

                                if layer_idx == 0:
                                    layer_prob_word = logits_selected
                                else:
                                    layer_prob_word = torch.vstack((layer_prob_word, logits_selected))
                                #print('add',layer_prob_word.size())


                            ###某一个sample的所有layer结果
                            filtered_tensor, filtered_words = process_logits(layer_prob_word, tokens_text)
                            layer_prob_word_f = filtered_tensor
                            token_first_word_modified.append(filtered_words)
                            layer_prob_word_f = layer_prob_word_f.cpu().numpy()
                            logits_tmp_word_modified.append(layer_prob_word_f)

                            layer_prob_first = layer_prob_word.cpu().numpy()
                            logits_tmp_modified.append(layer_prob_first)
                            
                        ###修改后结果
                        if args.cd == False:
                            for layer_idx in range(outputs.logits.size(0)):
                                last_layer_prob_each = [tensor.flatten()[0].item() for tensor in last_layer_prob_modified[layer_idx]]
                                last_layer_prob_soerted_each = sorted(last_layer_prob_each, reverse=True)
                                max1_each = last_layer_prob_soerted_each[0]
                                max1_idx_each = last_layer_prob_each.index(max1_each)
                                if answer_letter == max1_idx_each:
                                    layer_acc_modified[layer_idx] += 1
                        else:
                        ### ori +(ori-modified)
                            cd_logits = []
                            for a, b in zip(logits_tmp, logits_tmp_modified):
                                # 2. 校验每个数组的形状
                                if a.shape != b.shape:
                                    raise ValueError(f"数组形状不匹配: {a.shape} vs {b.shape}")
                                
                                # 3. 提升计算精度防止溢出（可选）
                                a_float32 = a.astype(np.float32)
                                b_float32 = b.astype(np.float32)
                                
                                # 4. 执行运算并保留原始数据类型
                                result_arr = (2 * a_float32 - b_float32).astype(a.dtype)
                                cd_logits.append(result_arr)
                                
                            for layer_idx in range(outputs.logits.size(0)):
                                last_layer_prob_each = [tensor.flatten()[0].item() for tensor in last_layer_prob_modified[layer_idx]]
                                #print('m', last_layer_prob_each)
                                last_layer_prob_each_ori = [tensor.flatten()[0].item() for tensor in last_layer_prob_ori[layer_idx]]
                                #print('o', last_layer_prob_each_ori)
                                modified_prob_each = [2*x - y for x, y in zip(last_layer_prob_each_ori, last_layer_prob_each)]
                                #print('f',modified_prob_each)
                                last_layer_prob_soerted_each = sorted(modified_prob_each, reverse=True)
                                max1_each = last_layer_prob_soerted_each[0]
                                max1_idx_each = modified_prob_each.index(max1_each)
                                if answer_letter == max1_idx_each:
                                    layer_acc_modified[layer_idx] += 1
                                
                        ###原模型选择的前两个
                        last_layer_prob_modified_unchanged=last_layer_prob_modified.copy()
                        modified_score  = [sublist[answer_letter] for sublist in last_layer_prob_modified_unchanged]
                        last_layer_prob_modified = [tensor.flatten()[0].item() for tensor in last_layer_prob_modified[-1]]
                        last_layer_prob_soerted = sorted(last_layer_prob_modified, reverse=True)
                        max1 = last_layer_prob_soerted[0]
                        max2 = last_layer_prob_soerted[1]
                        max1_idx = last_layer_prob_modified.index(max1)
                        max2_idx = last_layer_prob_modified.index(max2)
                        ori_token_first_sample = token_first_sample.copy()
                        ori_token_first_sample_false = token_first_sample_false.copy()
                        if answer_letter in [max1_idx, max2_idx]:
                            other_idx = max2_idx if answer_letter == max1_idx else max1_idx
                        else:
                            other_idx = max1_idx
                            
                        modified_score_false  = [sublist[other_idx] for sublist in last_layer_prob_modified_unchanged]
                        if args.cd_change == True:
                            if cd_score_change_r != None:
                                tmp_cd_score_change_r = [(m - o) / (abs(o) +1e-10) for m, o in zip(modified_score, ori_score)]
                                cd_score_change_r = [m + o for m, o in zip(cd_score_change_r, tmp_cd_score_change_r)]
                                tmp_cd_score_change = [(m - o) for m, o in zip(modified_score, ori_score)]
                                cd_score_change = [m + o for m, o in zip(cd_score_change, tmp_cd_score_change)]
                                
                                tmp_cd_score_change_r_false = [(m - o) / (abs(o)+1e-10) for m, o in zip(modified_score_false, ori_score_false)]
                                cd_score_change_r_false = [m + o for m, o in zip(cd_score_change_r_false, tmp_cd_score_change_r_false)]
                                tmp_cd_score_change_false = [(m - o) for m, o in zip(modified_score_false, ori_score_false)]
                                cd_score_change_false = [m + o for m, o in zip(cd_score_change_false, tmp_cd_score_change_false)]
                            else:
                                cd_score_change_r = [(m - o) / (abs(o)+1e-10) for m, o in zip(modified_score, ori_score)]
                                cd_score_change = [(m - o) for m, o in zip(modified_score, ori_score)]
                            
                                cd_score_change_r_false = [(m - o) / (abs(o)+1e-10) for m, o in zip(modified_score_false, ori_score_false)]
                                cd_score_change_false = [(m - o) for m, o in zip(modified_score_false, ori_score_false)]

                            
                        if args.cd == True:
                            logits_true_modified, logits_false_modified, token_first_sample, token_first_sample_false, diff_index = process_diff(token_first_modified, cd_logits, answer_letter, other_idx, b_idx, logits_true_modified, logits_false_modified)
                        else:
                            logits_true_modified, logits_false_modified, token_first_sample, token_first_sample_false, diff_index = process_diff(token_first_modified, logits_tmp_modified, answer_letter, other_idx, b_idx, logits_true_modified, logits_false_modified)
                        other_token = token_first_modified[other_idx][diff_index]
                        if args.cd_change == True:
                            if cd_logits_change is not None:
                                cd_logits_change+=(-ori_token_first_sample + token_first_sample)
                                cd_logits_change_r +=(-ori_token_first_sample + token_first_sample) / (abs(ori_token_first_sample)+1e-10)
                                cd_logits_change_false+=(-ori_token_first_sample_false + token_first_sample_false)
                                cd_logits_change_r_false +=(-ori_token_first_sample_false + token_first_sample_false) / (abs(ori_token_first_sample_false)+1e-10)
                            else:
                                cd_logits_change=(-ori_token_first_sample + token_first_sample)
                                cd_logits_change_r =(-ori_token_first_sample + token_first_sample) / (abs(ori_token_first_sample)+1e-10)
                                cd_logits_change_false=(-ori_token_first_sample_false + token_first_sample_false)
                                cd_logits_change_r_false =(-ori_token_first_sample_false + token_first_sample_false) / (abs(ori_token_first_sample_false)+1e-10)
                    ###单token logit变化绘图
                    #ax1.axhline(0, color='y', linestyle='--', linewidth=1, alpha=0.5)  # 新增参考线
                    ax1.plot(token_first_sample, marker='o', linestyle='-', color='b', label='True_token')
                    ax1.plot(token_first_sample_false, marker='o', linestyle='-', color='r', label='False_token')
                    ax1.plot(token_first_sample - token_first_sample_false, marker='o', linestyle='-', color='y', alpha=0.6,label='True-False')
                    
                    # 设置网格线（补充修改）
                    ax1.yaxis.set_major_locator(plt.MultipleLocator(5))  # 每5个单位一个主刻度
                    ax1.grid(axis='y', which='major', linestyle='--', linewidth=1, color='gray', alpha=0.6)
                    ax1.set_title("Logits of First Token")
                    ax1.set_xlabel("Layer Index")
                    #ax1.set_ylim(bottom=-2,top=16)
                    if args.logits_change_norm == True:
                        ax1.set_ylabel("Normalized Logits")
                    else:
                        ax1.set_ylabel("Logits Value")
                    
                    ###暂时不动
                    input_text = option[answer_letter]
                    input_encoding = tokenizer(input_text, return_tensors='pt').to(device)
                    attention_mask = input_encoding["attention_mask"]
                    input_id = input_encoding["input_ids"]
                    # 一个batch里答案开始到填充结束
                    answer_ids = input_id[:,prefix_id.shape[-1]:]

                    # attn+logits merge
                    with torch.no_grad():
                        outputs = model(input_ids=input_id, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True,num=args.num)
                    attentions = outputs.attentions
                    
                    #attentions_ori = outputs.attentions_ori
                    layer_prob = None
                    layer_attn = None
                    tokens_text = []
                    tokens = answer_ids.tolist()
                    query_tokens = prefix_id.tolist()

                    tokens_text = tokenizer.convert_ids_to_tokens(tokens[0], skip_special_tokens=True)
                    query_text2 = tokenizer.convert_ids_to_tokens(query_tokens[0], skip_special_tokens=False)
                    concept_token = tokens_text[diff_index]
                    query_text_tmp = query_text2.copy()
                    
                    if diff_index != 0:
                        for ans_idx in range(diff_index):
                            query_text_tmp.append(tokens_text[ans_idx])
                    for layer_idx in range(outputs.logits.size(0)):
                        num = len(answer_ids[0])
                        start_index = prefix_id.shape[-1] - 1
                        logits_selected = outputs.logits[layer_idx][start_index:start_index + num]
                        logits_selected = logits_selected[torch.arange(num), answer_ids[0]]
                        attn = attentions[layer_idx].mean(dim=1)[0]
                        attn_first = attn[start_index+diff_index, :prefix_id.shape[-1]+diff_index]
                        #print('get',attn_first.size())
                        if layer_idx == 0:
                            layer_attn=attn_first
                        else:
                            layer_attn = torch.vstack((layer_attn, attn_first))
                    filtered_tensor, filtered_words, p, cn, ca, cp, nc, p_idx, cn_idx, ca_idx, cv_idx, nc_idx = process_attn(layer_attn, query_text_tmp, token_word)
                    processed = []
                    for t in [p, cn, ca, cp, nc]:
                        if t == None:
                            zero = torch.zeros(32,1).to(device)
                            processed.append(zero)
                        else:
                            processed.append(t)
                    
                    concept = processed[1] + processed[2] + processed[3]
                    ### 不对比
                    if Concept != None:
                        Concept+=concept
                        No_concept += processed[-1]
                        P += processed[0]
                        Concept_N += processed[1]
                    else:
                        Concept=concept
                        No_concept = processed[-1]
                        P = processed[0]
                        Concept_N = processed[1]
                    if args.single_concept==True:
                        if args.attn_manipulation == True:
                            if args.cd == True:
                                if not os.path.exists(f"/data1/zjz/mmlu/res/special_case/cd_layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/Concept_attn"):
                                        os.makedirs(f"/data1/zjz/mmlu/res/special_case/cd_layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/Concept_attn")
                            else:
                                if not os.path.exists(f"/data1/zjz/mmlu/res/special_case/layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/Concept_attn"):
                                        os.makedirs(f"/data1/zjz/mmlu/res/special_case/layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/Concept_attn")
                        else:
                             if not os.path.exists(f"/data1/zjz/mmlu/res/special_case/ori/{args.subject}/Concept_attn"):
                                    os.makedirs(f"/data1/zjz/mmlu/res/special_case/ori/{args.subject}/Concept_attn")
                        layers = np.arange(len(model.model.layers))
                        concept_ratios_single = [concept[lay_idx].item() for lay_idx in layers]
                        p_ratios_single = [processed[0][lay_idx].item() for lay_idx in layers]
                        nc_ratios_single = [processed[-1][lay_idx].item() for lay_idx in layers]
                        concept_n_ratios_single = [processed[1][lay_idx].item() for lay_idx in layers]

                        # 创建画布
                        plt.figure(figsize=(10, 6))
                        
                        # 绘制三条折线
                        plt.plot(layers, concept_ratios_single, marker='o', label='Concept Ratio')
                        plt.plot(layers, p_ratios_single, marker='s', label='P Ratio')
                        plt.plot(layers, nc_ratios_single, marker='^', label='No Concept Ratio')
                        plt.plot(layers, concept_n_ratios_single, marker='x', label='Concept_N Ratio', alpha=0.5)
                        
                        # 添加标签和标题
                        plt.xlabel('Layer Index')
                        plt.ylabel('Ratio')
                        plt.title(f'Q {b_idx} Concept Ratio')
                        plt.legend()
                        plt.grid(True)
                        
                        # 保存图片
                        if args.attn_manipulation == True:
                            if args.cd == True:
                                plt.savefig(f"/data1/zjz/mmlu/res/special_case/cd_layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/Concept_attn/Q{b_idx}_{concept_token}.png", dpi=300, bbox_inches='tight')
                            else:
                                plt.savefig(f"/data1/zjz/mmlu/res/special_case/layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/Concept_attn/Q{b_idx}_{concept_token}.png", dpi=300, bbox_inches='tight')
                        else:
                            plt.savefig(f"/data1/zjz/mmlu/res/special_case/ori/{args.subject}/Concept_attn/Q{b_idx}_{concept_token}.png", dpi=300, bbox_inches='tight')
                        plt.close()

                    if layer_attn is not None:
                        #print(layer_attn.cpu().detach().numpy())
                        if args.attn_manipulation == True:
                            if args.cd == True:
                                    if not os.path.exists(f"/data1/zjz/mmlu/res/special_case/cd_layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}"):
                                        os.makedirs(f"/data1/zjz/mmlu/res/special_case/cd_layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}")
                            else:
                                if not os.path.exists(f"/data1/zjz/mmlu/res/special_case/layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}"):
                                    os.makedirs(f"/data1/zjz/mmlu/res/special_case/layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}")
                        else:
                            if not os.path.exists(f"/data1/zjz/mmlu/res/special_case/ori/{args.subject}"):
                                os.makedirs(f"/data1/zjz/mmlu/res/special_case/ori/{args.subject}")
                        num_layers = layer_attn.size(0)
                        layer_labels = [f"Layer {i}" for i in range(num_layers)]
                        token_labels = query_text_tmp
                        
                        sns.heatmap(
                            layer_attn.cpu().detach().numpy(),
                            annot=True,
                            fmt=".3f",
                            xticklabels=token_labels,
                            yticklabels=layer_labels,
                            cmap="coolwarm",
                            vmin=0,
                            vmax=0.1,
                            # vmin=-15,
                            # vmax=5,
                            cbar_kws={'label': 'Proportion'},
                            ax=ax0
                        )
                        ax0.xaxis.tick_top()
                        ax0.xaxis.set_label_position('top')
                        ax0.set_title(f"Correct token: {concept_token}, Contrastive Token: {other_token}", fontsize=14, fontweight='bold', pad=5)                 # 调整标题与图的间距
                        #ax0.set_xlabel("Token")
                        ax0.set_ylabel("Layer Index")
                        ax0.tick_params(axis='x', labelsize=12, rotation=30)           # 可选：倾斜角度防止重叠

                    # 添加标签和标题
                    if args.logits_change_norm == True:
                        plt.suptitle("Normalized Logits Variation")
                    else:
                        suptitle_text = f"Q{b_idx} Choose:{chr(65 + int(max1_idx))} Answer:{chr(65 + int(answer_letter))}"
                        plt.suptitle(suptitle_text, fontsize=16, fontweight='bold', y=0.97, verticalalignment='bottom') # 向下对齐
                    plt.legend()
                    if args.attn_manipulation == True:
                        if args.cd == True:
                            plt.savefig(f"/data1/zjz/mmlu/res/special_case/cd_layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/{args.subject}_Q{b_idx}_{concept_token}.png")
                        else:
                            plt.savefig(f"/data1/zjz/mmlu/res/special_case/layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/{args.subject}_Q{b_idx}_{concept_token}.png")
                    else:
                        plt.savefig(f"/data1/zjz/mmlu/res/special_case/ori/{args.subject}/{args.subject}_Q{b_idx}_{concept_token}.png")
                    plt.close()
                    
        if args.concept_token == True:
            # 生成带缩进的 JSON 字符串
            json_str = json.dumps(
                token_word,
                indent=2,
                ensure_ascii=False,
                default=set_encoder  # 关键参数
            )

            # 在逗号后添加换行（增强可读性）
            formatted_str = re.sub(r",(\s+)", ",\n\\1", json_str)

            with open(f"/data1/zjz/mmlu/res/special_case/concept_token//{args.subject}.json", "w", encoding="utf-8") as f:
                f.write(formatted_str)
                    
        if args.logits_change == True:
            if args.attn_manipulation == True:
                logits_true_modified = [x / total_num for x in logits_true_modified]
                logits_false_modified = [x / total_num for x in logits_false_modified]
            else: 
                logits_true_modified = [x / total_num for x in logits_true]
                logits_false_modified = [x / total_num for x in logits_false]
            # logits_true_word = [x / num_false for x in logits_true_word]
            # # logits_false_word = [x / total_num for x in logits_false_word]
            # logits_false_word = [x / num_false for x in logits_false_word]

            # if not os.path.exists(f"/root/zhujingze/mmlu/new_res/all_sample/logits_change_tf/{args.subject}"):
            #     os.makedirs(f"/root/zhujingze/mmlu/new_res/all_sample/logits_change_tf/{args.subject}")

            fig, ax = plt.subplots(figsize=(8, 5))
            #fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            
            # ax.plot(logits_true_word, marker='o', linestyle='-', color='g', label='Wrong_True_token')
            # ax.plot(logits_false_word, marker='o', linestyle='-', color='y', label='Wrong_False_token')
            ax.plot(logits_false_modified, marker='o', linestyle='-', color='r', label='Wrong_token')
            ax.plot(logits_true_modified, marker='o', linestyle='-', color='b', label='Right_token')
            ax.set_title("First Token")
            #ax[0].set_title("First Token")
            ax.set_xlabel("Layer Index")
            if args.logits_change_norm == True:
                ax.set_ylabel("Normalized Logits")
            else:
                ax.set_ylabel("Normalized Logits")

            # ax[1].plot(logits_true_word, marker='o', linestyle='-', color='b', label='True_token')
            # ax[1].plot(logits_false_word, marker='o', linestyle='-', color='r', label='False_token')
            # ax[1].set_title("Wrong First Token")
            # ax[1].set_xlabel("Layer Index")
            # if args.logits_change_norm == True:
            #     ax[1].set_ylabel("Normalized Logits")
            # else:
            #     ax[1].set_ylabel("Logits")

            # 添加标签和标题
            if args.logits_change_norm == True:
                plt.suptitle("Normalized Logits Variation")
            else:
                plt.suptitle("Normalized Logits Variation")
            plt.legend()
            if args.attn_manipulation == True:
                if args.cd == True:
                    plt.savefig(f"/data1/zjz/mmlu/res/special_case/cd_layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/logits_norm.png")
                else:
                    plt.savefig(f"/data1/zjz/mmlu/res/special_case/layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/logits_norm.png")
            else:
                plt.savefig(f"/data1/zjz/mmlu/res/special_case/ori/{args.subject}/norm_logits.png")
            plt.close() 

        if args.attn_concept == True:
            # 提取数据
            layers = np.arange(len(model.model.layers))
            concept_ratios = [Concept[lay_idx].item() / total_num for lay_idx in layers]
            p_ratios = [P[lay_idx].item() / total_num for lay_idx in layers]
            nc_ratios = [No_concept[lay_idx].item() / total_num for lay_idx in layers]
            concept_n_ratios = [Concept_N[lay_idx].item() / total_num for lay_idx in layers]

            # 创建画布
            plt.figure(figsize=(10, 6))
            
            # 绘制三条折线
            plt.plot(layers, concept_ratios, marker='o', label='Concept Ratio')
            plt.plot(layers, p_ratios, marker='s', label='P Ratio')
            plt.plot(layers, nc_ratios, marker='^', label='No Concept Ratio')
            plt.plot(layers, concept_n_ratios, marker='x', label='Concept_N Ratio', alpha=0.5)
            
            # 添加标签和标题
            plt.xlabel('Layer Index')
            plt.ylabel('Ratio')
            plt.title(f'{args.subject} Concept Ratio')
            plt.legend()
            plt.grid(True)
            
            # 保存图片
            if args.attn_manipulation == True:
                if args.cd == True:
                    plt.savefig(f"/data1/zjz/mmlu/res/special_case/cd_layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/concept_metrics.png", dpi=300, bbox_inches='tight')
                else:
                    plt.savefig(f"/data1/zjz/mmlu/res/special_case/layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/concept_metrics.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f"/data1/zjz/mmlu/res/special_case/ori/{args.subject}/concept_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
            # #print(f'True: {num_true} False: {num_false}')
            # for lay_idx in range(Concept.size(0)):
            #     print(f'All Layer {lay_idx} Concept: {Concept[lay_idx].item() / total_num:.2f} P:{P[lay_idx].item() / total_num:.2f} Nc:{No_concept[lay_idx].item() / total_num:.2f}')
            #     #print(f'True Layer {lay_idx + 1} Concept: {Concept[lay_idx].item() / num_true:.2f} P:{P[lay_idx].item() / num_true:.2f} Nc:{No_concept[lay_idx].item() / num_true:.2f}')
            # # for lay_idx in range(Concept.size(0)):
            # #     print(f'False Layer {lay_idx + 1} Concept: {Conceptf[lay_idx].item() / num_false:.2f} P:{Pf[lay_idx].item() / num_false:.2f} Nc:{No_conceptf[lay_idx].item() / num_false:.2f}')
                
        if args.output_layer_acc == True:
            # 提取数据
            layers = np.arange(len(model.model.layers))
            if args.attn_manipulation:
                accuracies = [layer_acc_modified[lay_idx] / total_num * 100 for lay_idx in layers]
            else:
                accuracies = [layer_acc[lay_idx] / total_num * 100 for lay_idx in layers]

            # 创建画布
            plt.figure(figsize=(10, 6))
            
            # 绘制折线
            plt.plot(layers, accuracies, marker='o', color='purple', linestyle='--')
            
            # 添加标签和标题
            plt.xlabel('Layer Index')
            plt.ylabel('Accuracy (%)')
            plt.title(f'{args.subject} Accuracy')
            plt.grid(True)
            
            # 保存图片
            if args.attn_manipulation == True:
                if args.cd == True:
                     plt.savefig(f"/data1/zjz/mmlu/res/special_case/cd_layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/layer_accuracy.png", dpi=300, bbox_inches='tight')
                else:
                    plt.savefig(f"/data1/zjz/mmlu/res/special_case/layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}/{args.subject}/layer_accuracy.png", dpi=300, bbox_inches='tight')
            else:
                 plt.savefig(f"/data1/zjz/mmlu/res/special_case/ori/{args.subject}/layer_accuracy.png", dpi=300, bbox_inches='tight')
            plt.close()
            if args.attn_manipulation == True:
                if args.cd ==True:
                    output_dir = f"/data1/zjz/mmlu/res/special_case/layer_acc/cd_layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}"
                else:
                    output_dir = f"/data1/zjz/mmlu/res/special_case/layer_acc/layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}"
            else:
                output_dir = f"/data1/zjz/mmlu/res/special_case/layer_acc/ori"
            os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
            output_path = f"{output_dir}/{args.subject}.txt"

            # 写入文件并打印到控制台
            with open(output_path, 'w') as f:  # 使用 'w' 模式会覆盖原有内容，如需追加可改为 'a'
                for lay_idx in range(len(model.model.layers)):
                    # 根据条件选择正确的准确率数据
                    if args.attn_manipulation:
                        accuracy = layer_acc_modified[lay_idx] / total_num * 100
                    else:
                        accuracy = layer_acc[lay_idx] / total_num * 100
                    
                    # 格式化输出内容
                    line = f'Layer {lay_idx}, {accuracy:.2f}'
                    
                    # 同时执行打印和文件写入
                    f.write(line + '\n')  # 写入换行符到文件
                    
            if args.cd_change:
                output_path_cd = f"/data1/zjz/mmlu/res/special_case/logits_change/layer_{args.start_layer}-{args.end_layer}_en_{args.token_enhance}_wk_{args.token_weaken}_a_{args.alpha}"
                os.makedirs(output_path_cd, exist_ok=True)  # 确保目录存在
                output_path_file = f"{output_path_cd}/{args.subject}.txt"
                with open(output_path_file, 'w') as f:  # 使用 'w' 模式会覆盖原有内容，如需追加可改为 'a'
                    for lay_idx in range(len(model.model.layers)):
                        # 根据条件选择正确的准确率数据
                        logits_r = cd_logits_change_r[lay_idx] / total_num
                        logits_d = cd_logits_change[lay_idx] / total_num
                        
                        score_r = cd_score_change_r[lay_idx] / total_num
                        score_d = cd_score_change[lay_idx] / total_num
                        
                        logits_r_m = (cd_logits_change_r[lay_idx] - cd_logits_change_r_false[lay_idx])/ total_num
                        logits_d_m = (cd_logits_change[lay_idx] - cd_logits_change_false[lay_idx]) / total_num
                        
                        score_r_m = (cd_score_change_r[lay_idx] - cd_score_change_r_false[lay_idx]) / total_num
                        score_d_m = (cd_score_change[lay_idx] - cd_score_change_false[lay_idx]) / total_num
                        #score_change = ori_score modified_score
                        
                        # 格式化输出内容
                        line = f'{lay_idx}, Logits d:{logits_d:.4f}, Logits r:{logits_r:.4f}, Score d:{score_d:.4f}, Scores r:{score_r:.4f}, Logits dm:{logits_d_m:.4f}, Logits rm:{logits_r_m:.4f}, Score dm:{score_d_m:.4f}, Scores rm:{score_r_m:.4f}'
                        
                        # 同时执行打印和文件写入
                        f.write(line + '\n')  # 写入换行符到文件

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--subject', type=str)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--method', type=str)
    parser.add_argument('--num', type=int)
    parser.add_argument('--attn_manipulation', action='store_true', default=False)
    parser.add_argument('--output_layer_acc', action='store_true', default=False)
    parser.add_argument('--concept_token', action='store_true', default=False)
    parser.add_argument('--attn_concept', action='store_true', default=False)
    parser.add_argument('--logits_change_norm', action='store_true', default=False)
    parser.add_argument('--logits_change', action='store_true', default=False)
    parser.add_argument('--logits_sample', action='store_true', default=False)
    parser.add_argument('--cd', action='store_true', default=False)
    parser.add_argument('--cd_change', action='store_true', default=False)
    parser.add_argument('--single_concept', action='store_true', default=False)
    parser.add_argument('--attn', type=bool)
    parser.add_argument('--start_layer', type=int)
    parser.add_argument('--end_layer', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--token_enhance', type=str)
    parser.add_argument('--token_weaken', type=str)


    args = parser.parse_args()
    main(args)
