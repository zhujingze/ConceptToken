# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re, pdb
import os
import json
import numpy as np
import transformers
from tqdm import tqdm
import argparse
import ssl
import urllib.request
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve
import time
from utils.utils_know import *
from sled_decoding_gen import SLED_DecodedLLM_Know as SLED_DecodedLLM
import json
import warnings

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

# N_SHOT = 7
# COT_FLAG = True
# ANSWER_TRIGGER = "So the answer is"

def load_csv(dataset_name, debug):
    # input file is in csv format, can be loaded by pandas
    # required columns: [prompt] only
    
    if dataset_name == 'triviaqa':
        dataset = load_dataset("parquet",data_files={'validation':'/root/zhujingze/Concept/benchmark/triqa/unfiltered.nocontext/validation-00000-of-00001.parquet'})
    elif dataset_name == 'natural_questions':
        dataset = load_dataset("parquet",data_files={'validation':'/root/zhujingze/Concept/benchmark/nq/nq_open/validation-00000-of-00001.parquet'})
    elif dataset_name == 'hotpotqa':
        local_files={"validation": "/root/zhujingze/Concept/benchmark/hot/hotpot_dev_fullwiki_v1.json"}
        dataset = load_dataset(path="/root/zhujingze/Concept/hotpot_qa.py",name="distractor",data_files=local_files,trust_remote_code=True)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")      
    dataset=dataset['validation']
    list_data = list(dataset['question'])
    labels = list(dataset['answer'])
    
    if debug:
        list_data = list_data[0:20]
        labels = labels[0:20]

    return list_data,labels

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

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
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer




def build_prompt(question_text, prompt_style='zero_shot'):
    # this prompt is designed for trivia QA
    if prompt_style == 'zero_shot':
        question_text_prompt = 'Answer the following question concisely.\n'
        question_text_prompt += f'Q:{question_text}\nA:'
    elif prompt_style == 'few_shot':
        # question_text_prompt = 'Answer the following question concisely.\n'
        question_text_prompt = f'Q: Who was President when the first Peanuts cartoon was published?\nA: Harry Truman\n\n'
        # question_text_prompt += f'Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?\nA: Sinclair Lewis\n\n'
        question_text_prompt += f'Q: Where in England was Dame Judi Dench born?\nA: York\n\n'
        question_text_prompt += f'Q: {question_text}\nA: '
    elif prompt_style == 'zero_shot_w_instru':
        raise NotImplementedError("zero_shot_w_instru Not implemented yet.")
    return question_text_prompt

def plot_auroc_scores(is_correct_list, scores_list, output_file, method_name):
    
    # Separate scores into correct and incorrect
    correct_scores = [score for is_correct, score in zip(is_correct_list, scores_list) if is_correct]
    incorrect_scores = [score for is_correct, score in zip(is_correct_list, scores_list) if not is_correct]

    # check if correct_scores and incorrect_scores are nan
    if np.isnan(correct_scores).any() or np.isnan(incorrect_scores).any():
        print(f"Error: there is nan, skip computing AUROC, AUPRC, AURC for {method_name}")
        auroc = None
        auprc = None
        aurc = None
        scores = {'auroc': auroc, 'auprc': auprc, 'aurc': aurc}
        return scores
    
    y_true = [1]*len(correct_scores) + [0]*len(incorrect_scores)
    y_scores = correct_scores + incorrect_scores

    
    # Compute AUROC
    auroc = roc_auc_score(y_true, y_scores)

    # Compute AUPRC
    auprc = average_precision_score(y_true, y_scores)

    # Compute AURC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aurc = auc(recall, precision)


    # Create the plot
    plt.figure()
    plt.hist(correct_scores, bins=20, alpha=0.5, label='Correct')
    plt.hist(incorrect_scores, bins=20, alpha=0.5, label='Incorrect')
    plt.legend(loc='upper right')
    plt.title(f'AUROC: {auroc:.2f}')
    
    # Save the plot
    output_dir = os.path.dirname(output_file)
    plt.savefig(os.path.join(output_dir, f'detect_{method_name}_plot.png'))
    plt.close()
    
    scores = {'auroc': auroc, 'auprc': auprc, 'aurc': aurc}
    return scores

if __name__ == "__main__":
    start=time.time()
    parser = argparse.ArgumentParser()
    # parser.add_argument("--val_test_mode", type=str, default="1")
    
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--dataset_name", type=str, choices=["triviaqa", "natural_questions", "hotpotqa"], default="triviaqa")
    parser.add_argument("--prompt_style", type=str, choices=["zero_shot", "few_shot", "zero_shot_w_instru"], default='few_shot')
    ###########
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num_gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data_path", type=str, default="Data/StrategyQA")
    parser.add_argument("--output_path", type=str, default="./strqa_result")
    parser.add_argument("--output_file", type=str, default="./strqa_result")
    parser.add_argument("--early-exit-layers", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--decoding_method", type=str, default="VanillaGreedy", choices=["VanillaGreedy", "SLED", "dola","attn"])
    parser.add_argument("--evolution_rate", type=float, default=2)
    parser.add_argument("--evolution_scale", type=int, default=10)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--start_layer", type=int)
    parser.add_argument("--end_layer", type=int)
    parser.add_argument("--attn_alpha", type=float)
    parser.add_argument("--token_enhance", type=str)
    parser.add_argument("--token_weaken", type=str)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--sink", type=bool)
    parser.add_argument("--th", type=float)
    parser.add_argument("--ema", type=bool)
    parser.add_argument("--single", type=bool)
    parser.add_argument("--ave", type=bool)
    parser.add_argument("--including_answers", type=bool)
    parser.add_argument("--sink_layers",
                   type=lambda s: [int(x) for x in s.split(',')],
                   default=[],
                   help="要启用的层号列表，用逗号分隔（例如：'1,3,5'）")

    
    args = parser.parse_args()
    output_file = args.output_file
    model_name = args.model_name
    num_gpus = args.num_gpus
    output_path=args.output_path
    device = args.device
    start_layer=args.start_layer
    end_layer=args.end_layer
    attn_alpha=args.attn_alpha
    token_enhance=args.token_enhance
    token_weaken=args.token_weaken
    beta = args.beta
    sink = args.sink
    sink_layers = args.sink_layers
    th = args.th
    ema = args.ema
    single = args.single
    ave = args.ave
    including_answers = args.including_answers    
    model_name_input = os.path.basename(model_name.rstrip('/'))
    llm = SLED_DecodedLLM(model_name, device, num_gpus, args.max_gpu_memory)
    if args.decoding_method in ["VanillaGreedy", "attn"]:
        if args.early_exit_layers is not None:
            warnings.warn("The 'early_exit_layers' argument should be None when using Vanilla greedy decoding.")
        print("Vanilla greedy decoding from the final layer", flush=True)
        mature_layer = None
        candidate_premature_layers = None
        early_exit_layers = [-1]

    else:
        if args.early_exit_layers is None:
            early_exit_layers = [int(x) for x in range(llm.num_layers + 1)]
        else:
            early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]

        print(f"MODE: {args.decoding_method} decoding with the final layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mature_layer = early_exit_layers[-1]
        candidate_premature_layers = early_exit_layers[:-1]
         
    # load dataset
    list_data_dict,labels = load_csv(args.dataset_name, args.debug)
    
    # if args.parallel:
    #     chunk_size = len(list_data_dict) // args.total_shard
    #     list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    
    # stop_word_list = ["Q:", "\n\n##"]
    stop_word_list = ["Q:"]
    llm.set_stop_words(stop_word_list)
    
    generate_kwargs = dict(single=single,ave=ave,model_name_input=model_name_input,including_answers=including_answers,
                th=th,ema=ema,sink_layers=sink_layers,sink=sink,beta=beta,token_weaken=token_weaken,
                token_enhance=token_enhance,attn_alpha=attn_alpha,start_layer=start_layer,end_layer=end_layer,
                max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k,
                temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=args.decoding_method, 
                mature_layer=mature_layer, candidate_premature_layers=candidate_premature_layers, 
                relative_top=args.relative_top,relative_top_value=args.relative_top_value,evolution_rate=args.evolution_rate,
                evolution_scale=args.evolution_scale
            )
        
    result_dict = {'qid_list':[], 'answers': {}, 'model_completion': {}, 'questions': {}, 'logit_scores': {}}
    
    print("Begin inference...\n")
    # print("***Hyperparameters***:", args)
    print("\nSample prompt: \n", build_prompt(list_data_dict[0], args.prompt_style))
    print("*"*20)
    print("\n\n")
    
    os.makedirs(args.data_path, exist_ok=True) 

    try:
        permute_idx = np.load(os.path.join(args.data_path, "val_test_idx_{}.npy"))
    except:
        permute_idx = np.random.permutation(len(list_data_dict))  
        np.save(os.path.join(args.data_path, "val_test_idx_{}.npy"), permute_idx)

    # val_idx = permute_idx[0:100]
    # test_idx = permute_idx[100:]

    # val_idx = permute_idx[0:int(len(list_data_dict)*.2)]
    # test_idx = permute_idx[int(len(list_data_dict)*.2):]
    
    # val_dataset = [list_data_dict[i] for i in val_idx]
    # test_dataset = [list_data_dict[idx] for idx in test_idx]

    # val_label = [labels[i] for i in val_idx]
    # test_label = [labels[idx] for idx in test_idx]
    # dataset=list_data_dict
    # if args.val_test_mode=='val':
    #     dataset=val_dataset
    #     labels=val_label
    # elif args.val_test_mode=='test':
    #     dataset=test_dataset
    #     labels=test_label
    
    dataset=list_data_dict
    # dataset=dataset[:10]
    # labels=labels[:10]

    for i, question in enumerate(tqdm(dataset)):
    # for i, question in enumerate(tqdm(val_dataset, desc='Processing')):

        answer=labels[i]
        prompt=build_prompt(question, args.prompt_style)

        # if args.return_adjust_scores:
        #     model_completion, c_dist, outputs = llm.generate(prompt, **generate_kwargs)
        #     # logit_scores = llm.get_lm_scores_from_outputs(outputs, mode=mode)
        # else:
        model_completion, c_dist = llm.generate(prompt, **generate_kwargs)
        # pdb.set_trace()
        logit_scores=0
        # if mode=='baseline' or mode=='dola' or mode=='with_dola':
        #     logit_scores=0
        # else:
        #     logit_scores = llm.get_lm_scores_from_outputs(outputs, mode=mode)

        # process output format to remove unnecessary tokens; designed for few-shot prompt
        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]
        if 'Q:' in model_completion:
            model_completion = model_completion.split('Q:')[0].strip()
        model_completion = model_completion.strip()

        # # TODO: what is this for?
        # if mode in ["dola", "activation"]:
        #     for k, v in c_dist.items():
        #         premature_layer_dist[k] += v
                        
        print("-"*20)
        #print(f"Q{i}: {question}\nA: {answer}\nModel Response after processing: {model_completion}\n\n")
        
        result_dict['qid_list'].append(i)
        result_dict['answers'][i] = answer
        result_dict['model_completion'][i] = model_completion
        result_dict['questions'][i] = question
        result_dict['logit_scores'][i] = logit_scores
        
        if args.debug:
            if i > 10:
                break     
        
        
        # here I note the next 'print' lines
    '''
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        

    
        print(f'Question: {sample}\n\n'
            f'Model Completion: {model_completion}\n\n')

        print(f'Num of total question: {len(answers)}.')
    if mode == "dola" or mode=="activation" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:
                
                print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))
    '''
    
    # end=time.time()
    # print(f"time:{end-start}s")
    # pdb.set_trace()
    # save results to a json file
    # model_tag = "llama-7b" from model_name "huggyllama/llama-7b"
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    print(f"Saving results to {output_path}")
    print("Begin evaluation...")

    # evaluation
    if args.do_rating:
        ground_truth = result_dict['answers']
        predicted_answers = result_dict['model_completion']
        qid_list = result_dict['qid_list']
        if args.dataset_name in ['triviaqa', 'hotpotqa']:
            eval_metrics = evaluate_triviaqa(ground_truth, predicted_answers, qid_list=qid_list, mute=False)
        elif args.dataset_name == 'natural_questions':
            eval_metrics = evaluate_nq(ground_truth, predicted_answers, qid_list=qid_list, mute=False)
        else:
            raise NotImplementedError(f"Dataset {args.dataset_name} not implemented yet.")
        
        # remove 'error_id' from eval_metrics
        if 'error_id' in eval_metrics:
            error_id_list = eval_metrics['error_id']
            del eval_metrics['error_id']
            eval_metrics['num_error'] = len(error_id_list)
            
            error_samples = {}
            for id in error_id_list:
                question = result_dict['questions'][id]
                answer = result_dict['answers'][id]['normalized_aliases'] if args.dataset_name == 'triviaqa' else result_dict['answers'][id]
                prediction = result_dict['model_completion'][id]
                print(f"\n\nQ: {question}\nGT: {answer}\nA: {prediction}")
                error_sample = {'Q':question, 'model_prediction': prediction, 'A': answer, 'correct': 0}
                error_samples[id] = error_sample
                
            # record all the correct samples
            correct_samples = {}
            for id in qid_list:
                if id not in error_id_list:
                    question = result_dict['questions'][id]
                    answer = result_dict['answers'][id]['normalized_aliases'] if args.dataset_name == 'triviaqa' else result_dict['answers'][id]
                    prediction = result_dict['model_completion'][id]
                    # print(f"\n\nQ: {question}\nGT: {answer}\nA: {prediction}")
                    correct_sample = {'Q':question, 'model_prediction': prediction, 'A': answer, 'correct': 1}
                    correct_samples[id] = correct_sample

            final_samples = {'error_samples': error_samples, 'correct_samples': correct_samples}            
            with open(output_file.replace('.json', '_results.json'), 'w') as f:
                json.dump(final_samples, f)
                
        # if args.return_adjust_scores:
        # # compute auroc and plot the distribution of scores
        #     is_correct_list = [eval_metrics['is_correct'][i] for i in qid_list]
        #     score_names = next(iter(result_dict['logit_scores'].values())).keys()
        #     del eval_metrics['is_correct']
        #     if 'origin_log_prob' in score_names:
        #         origin_log_prob_list = np.array([result_dict['logit_scores'][id]['origin_log_prob'] for id in qid_list])
        #         origin_scores = plot_auroc_scores(is_correct_list, origin_log_prob_list, output_file, "origin_log_prob")
        #         eval_metrics['origin_log_prob'] = origin_scores
        #     if 'entropy' in score_names:
        #         entropy_list = np.array([result_dict['logit_scores'][id]['entropy'] for id in qid_list])
        #         entropy_scores = plot_auroc_scores(is_correct_list, entropy_list, output_file, "entropy")      
        #         eval_metrics['entropy'] = entropy_scores    
        #     if 'final_log_prob' in score_names:
        #         final_log_prob_list = np.array([result_dict['logit_scores'][id]['final_log_prob'] for id in qid_list])
        #         final_scores = plot_auroc_scores(is_correct_list, final_log_prob_list, output_file, "final_log_prob")
        #         eval_metrics['final_log_prob'] = final_scores
 
            
        exact_match_acc = eval_metrics['exact_match']
        f1 = eval_metrics['f1']
        print(f"acc:{exact_match_acc:.5f}\nf1:{f1:.5f}")
        
        # pdb.set_trace()
        eval_metrics['model_name'] = model_name
        eval_metrics['dataset'] = 'triviaqa'
        eval_metrics['early_exit_layers'] = early_exit_layers
        #eval_metrics['mode'] = mode
        # save all the paramters of args into eval_metrics
        eval_metrics['parameters'] = vars(args)
        eval_metrics['sample_prompt'] = build_prompt(list_data_dict[0], args.prompt_style)
        with open(output_file.replace('.json', '_rating.json'), 'w') as f:
            json.dump(eval_metrics, f)
