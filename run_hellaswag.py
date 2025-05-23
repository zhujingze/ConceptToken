import re
import os
import json
import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange
import argparse
from utils.utils_truthfulqa import MC_calcs
import ssl
import urllib.request
from sled_decoding import SLED_DecodedLLM_HELLA as SLED_DecodedLLM
import json
import copy
from datasets import load_dataset
from itertools import combinations

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = True
ANSWER_TRIGGER = "The answer is"

def load_hellaswag(data_path=None, pondering=None, keys_path=None):
    if keys_path is not None:
        with open(keys_path, "r", encoding="utf-8") as f:
            key_words = json.load(f)
    
    if data_path:
        dataset = load_dataset('json',data_files={'validation':'/hellaswag/data/hellaswag_val.jsonl'})
    else:
        dataset = load_dataset("Rowan/hellaswag", trust_remote_code=True)
    print(dataset)
    data = dataset["validation"]
    cnt = 0
    list_data_dict = []
    completion_lens = []
    for idx, item in enumerate(data):
        
        # Convert label to integer if it is a string
        label = int(item["label"]) if isinstance(item["label"], str) else item["label"]

        # Exclude the correct label from contradictions
        contradictions = [ending for i, ending in enumerate(item["endings"]) if i != item["label"]]
        
        formatted_item = {
            "prefix": item['activity_label']+':'+item["ctx"],  # Combine ctx_a and ctx_b if needed
            "completion": item["endings"][label],
            "contradiction_0": contradictions[0],
            "contradiction_1": contradictions[1],
            "contradiction_2": contradictions[2],
        }
        
        correct_length = len(item["endings"][label])
        incorrect_lengths = [len(ending) for i, ending in enumerate(item["endings"]) if i != label]

        completion_lengths = [correct_length] + incorrect_lengths
        completion_lens.append(completion_lengths)
        
      
        list_data_dict.append(formatted_item)
    
    return list_data_dict


def download_url(url: str, folder='folder'):
    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="./results")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default=None)
    parser.add_argument("--post_softmax", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--relative_top", type=float, default=0.0)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--decoding_method", type=str, default="VanillaGreedy", choices=["VanillaGreedy","SLED", "dola", "attn"])
    parser.add_argument("--evolution_rate", type=float, default=2)
    parser.add_argument("--evolution_scale", type=int, default=10)
    parser.add_argument("--start_layer", type=int)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--end_layer", type=int)
    parser.add_argument("--attn_alpha", type=float)
    parser.add_argument("--token_enhance", type=str)
    parser.add_argument("--token_weaken", type=str)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--sink", type=bool)
    parser.add_argument("--ema", type=bool)
    parser.add_argument("--th", type=float)
    parser.add_argument("--single", type=bool, default=False)
    parser.add_argument("--sink_layers",
                   type=lambda s: [int(x) for x in s.split(',')],
                   default=[],
                   help="要启用的层号列表，用逗号分隔（例如：'1,3,5'）")
    
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    start_layer=args.start_layer
    end_layer=args.end_layer
    attn_alpha=args.attn_alpha
    token_enhance=args.token_enhance
    token_weaken=args.token_weaken
    beta = args.beta
    sink = args.sink
    sink_layers = args.sink_layers
    ema = args.ema
    single=args.single
    th = args.th
    model_name_input = os.path.basename(model_name.rstrip('/'))

    
    list_data_dict = load_hellaswag(args.data_path)
    output_file = args.output_path
   
    llm = SLED_DecodedLLM(model_name, device, num_gpus, args.max_gpu_memory)
    if args.decoding_method in ["VanillaGreedy", "attn"]:
        if args.early_exit_layers is not None:
            import warnings
            warnings.warn("The 'early_exit_layers' argument should be None when using Vanilla greedy decoding.")
        print("Vanilla greedy decoding from the final layer", flush=True)
        mature_layer = None
        candidate_premature_layers = None

    else:
        if args.early_exit_layers is None:
            early_exit_layers = [int(x) for x in range(llm.num_layers + 1)]
        else:
            early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
        print(
            f"MODE: {args.decoding_method} decoding with the final layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mature_layer = early_exit_layers[-1]
        candidate_premature_layers = early_exit_layers[:-1]
    ###mc
    result_dict = {'question': [], 'model_scores': [], 'total_mc1': 0.0, 'total_mc2': 0.0, 'total_mc3': 0.0}
    answers = []
    #result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    
    for idx in tqdm(range(len(list_data_dict))):
        sample = list_data_dict[idx]
        ###MCx
        scores_true = []
        scores_false = []

        token_ranges=None
        input_text_keys = None
        context = sample['prefix']
        answer_true = ' ' + sample['completion']
        answers_false = []
        for i in range(3):
            answers_false.append(' ' + sample[f'contradiction_{i}'])
        
        generate_kwargs = dict(do_sample=args.do_sample, mode=args.decoding_method, mature_layer=mature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top, relative_top_value=args.relative_top_value,evolution_rate=args.evolution_rate,evolution_scale=args.evolution_scale)
        completion_len = []
        answer_true_log_prob,correct_len,_= llm.lm_score(model_name_input,context, answer_true,single=single, start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta, sink=sink,sink_layers=sink_layers,ema=ema,th=th, **generate_kwargs)
        completion_len.append(correct_len)
        #answer_true_log_prob , _= llm.lm_score(model_name_input,context, answer_true,single=single, start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta, sink=sink,sink_layers=sink_layers,ema=ema,th=th, **generate_kwargs)
        ###MC
        scores_true.append(answer_true_log_prob / correct_len)

        answer_false_log_probs = []
        for answer_false in answers_false:
            answer_false_log_prob ,incorrect_len,_= llm.lm_score(model_name_input,context, answer_false, single=single,start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta,sink=sink,sink_layers=sink_layers,ema=ema,th=th, **generate_kwargs)
            completion_len.append(incorrect_len)
            #answer_false_log_prob , _= llm.lm_score(model_name_input,context, answer_false, single=single,start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta,sink=sink,sink_layers=sink_layers,ema=ema,th=th, **generate_kwargs)
            answer_false_log_probs.append(answer_false_log_prob)
            ###MC
            scores_false.append(answer_false_log_prob / incorrect_len)
       
        scores = MC_calcs(scores_true, scores_false, answer_true, answer_true)
        if np.isnan(scores['MC1']) or np.isnan(scores['MC2']) or np.isnan(scores['MC3']):
            import ipdb;

            ipdb.set_trace()

        result_dict['model_scores'].append(scores)
        result_dict['question'].append(sample)
        # update total scores
        result_dict['total_mc1'] += scores['MC1']
        result_dict['total_mc2'] += scores['MC2']
        result_dict['total_mc3'] += scores['MC3']


        is_cor = True
        log_probs = [answer_true_log_prob] + answer_false_log_probs 
        normalized_log_probs = log_probs / np.array(completion_len)
        #normalized_log_probs = log_probs
        predicted_answer_idx = np.argmax(normalized_log_probs)
        if predicted_answer_idx == 0: 
            is_cor = True
        else:
            is_cor = False
        
        answers.append(is_cor)
        # result_dict['is_correct'].append(is_cor)
        # result_dict['model_completion'].append([answer_true_log_prob] + answer_false_log_probs)

        # print(f'Num of total question: {len(answers)}, '
        #     f'correct num: {sum(answers)}, '
        #     f'correct rate: {float(sum(answers))/len(answers)}.')

    # with open(output_file, "a") as file:
    #     result_info = (f"Num of total questions: {len(answers)}, "
    #                 f"correct num: {sum(answers)}, "
    #                 f"correct rate: {float(sum(answers)) / len(answers):.5f}.")

    #     file.write(f"{result_info}\n")
    # print(f'Num of total question: {len(answers)}, '
    #     f'correct num: {sum(answers)}, '
    #     f'correct rate: {float(sum(answers))/len(answers)}.')
    result_dict['total_mc1'] /= len(result_dict['question'])
    result_dict['total_mc2'] /= len(result_dict['question'])
    result_dict['total_mc3'] /= len(result_dict['question'])

    #print(f'Final MC1/2/3: \n{result_dict["total_mc1"]}, {result_dict["total_mc2"]}, {result_dict["total_mc3"]}')
    print(f'ACC: {result_dict["total_mc1"]}')
