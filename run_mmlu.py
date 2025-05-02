# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/voidism/DoLa

import transformers
from tqdm import tqdm, trange
import argparse
from sled_decoding import SLED_DecodedLLM_MMLU as SLED_DecodedLLM
import json
import warnings
from dataset_mmlu_lens import prepare_mc_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from torch.utils.data import DataLoader

transformers.logging.set_verbosity(40)

import os
import re
from glob import glob

def get_prefix_list_advanced(folder_path):
    pattern = os.path.join(folder_path, "*_test.csv")
    files = sorted([f for f in glob(pattern) if os.path.isfile(f)])
    
    prefix_list = []
    for f in files:
        filename = os.path.basename(f)
        if re.match(r"^.+_test\.csv$", filename):  # 正则校验格式
            prefix_list.append(filename[:-9])  # 等价于 split("_test.csv")[0]
    
    return prefix_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data_path", type=str, default="Data/FACTOR/wiki_factor.csv")
    parser.add_argument("--output-path", type=str, default="./gsm8k_result")
    parser.add_argument("--early-exit-layers", type=str, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoding_method", type=str, default="VanillaGreedy", choices=["VanillaGreedy","SLED", "dola", "attn"])
    parser.add_argument("--evolution_rate", type=float, default=2)
    parser.add_argument("--evolution_scale", type=int, default=10)
    parser.add_argument("--start_layer", type=int)
    parser.add_argument("--end_layer", type=int)
    parser.add_argument("--attn_alpha", type=float)
    parser.add_argument("--token_enhance", type=str)
    parser.add_argument("--token_weaken", type=str)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--sink", type=bool)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--ema", type=bool)
    parser.add_argument("--th", type=float)
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
    th = args.th
    tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
    #list_data_dict = load_factor_dataset(args.data_path)
    dataset_all = []
    #print(sink)
    print(f'{start_layer}-{end_layer}_{attn_alpha}_{beta}_s{sink}')
    if args.subject != 'all':
        test_filename = args.subject + '_test.csv'
        val_filename = args.subject + '_val.csv'
        
        test_file = os.path.join(os.path.join(args.data_path, 'test'), test_filename)
        val_file = os.path.join(os.path.join(args.data_path, 'val'), val_filename)
        
        dataset = prepare_mc_dataset(
            subject=args.subject,
            csv_file=test_file,
            example_file=val_file,
            tokenizer=tokenizer
        )
        dataset_all.append(dataset)
    elif args.subject == 'all':
        
        subjects = get_prefix_list_advanced(os.path.join(args.data_path, 'test'))
        for subject in subjects:
            test_filename = subject + '_test.csv'
            val_filename = subject + '_val.csv'
            
            test_file = os.path.join(os.path.join(args.data_path, 'test'), test_filename)
            val_file = os.path.join(os.path.join(args.data_path, 'val'), val_filename)
            
            dataset = prepare_mc_dataset(
                subject=subject,
                csv_file=test_file,
                example_file=val_file,
                tokenizer=tokenizer
            )
            dataset_all.append(dataset)
    
    llm = SLED_DecodedLLM(model_name, device, num_gpus, args.max_gpu_memory)
    #llm.set_stop_words(["Q:", "\end{code}"])

    if args.decoding_method in ["VanillaGreedy", "attn"]:
        if args.early_exit_layers is not None:
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

        # 迭代使用数据
    for dataset in dataset_all:
        answers = []
        for sample in dataset:
            prefix = sample["prefix"]
            true_answer = sample["true_answer"]
            false_answers = sample['false_answers']
            subject = sample["subject"]
            
            generate_kwargs = dict(do_sample=args.do_sample, mode=args.decoding_method, mature_layer=mature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top, relative_top_value=args.relative_top_value,evolution_rate=args.evolution_rate,evolution_scale=args.evolution_scale)
            answer_true_log_prob, c_dist = llm.lm_score(prefix, true_answer, start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta, sink=sink,sink_layers=sink_layers,ema=ema,th=th, **generate_kwargs)

            answer_false_log_probs = []
            for answer_false in false_answers:
                #print(answer_false)
                answer_false_log_prob, c_dist = llm.lm_score(prefix, answer_false, start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta,sink=sink,sink_layers=sink_layers,ema=ema,th=th, **generate_kwargs)
                answer_false_log_probs.append(answer_false_log_prob)

            is_cor = True

            # print('T',answer_true_log_prob)
            # print('F',answer_false_log_probs)
            for answer_false_log_prob in answer_false_log_probs:
                if answer_true_log_prob < answer_false_log_prob:
                    is_cor = False
                    break

            answers.append(is_cor)
            # result_dict['is_correct'].append(is_cor)
            # result_dict['model_completion'].append([answer_true_log_prob] + answer_false_log_probs)

        print(f'{subject}: {float(sum(answers))/len(answers)}')
    # if 'wiki' in args.data_path:
    #     sub = 'wiki'
    # elif 'news' in args.data_path:
    #     sub = 'news'
    # elif 'expert' in args.data_path:
    #     sub = 'expert'
    # print(f'{args.decoding_method}_{sub}_{args.start_layer}-{args.end_layer}_{args.attn_alpha}_{args.beta}, correct rate: {float(sum(answers))/len(answers)}.')

    # # save results to a json file
    # model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    # with open(args.output_path, 'w') as f:
    #     json.dump(result_dict, f)
