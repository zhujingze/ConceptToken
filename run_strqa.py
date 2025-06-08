# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/voidism/DoLa

import transformers
from tqdm import tqdm, trange
import argparse
from utils.utils_strqa import *
from sled_decoding_gen import SLED_DecodedLLM_StrQA as SLED_DecodedLLM
import json
import warnings
transformers.logging.set_verbosity(40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num_gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data_path", type=str, default="Data/StrategyQA")
    parser.add_argument("--output_path", type=str, default="./strqa_result")
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
    th = args.th
    ema = args.ema
    single = args.single
    ave = args.ave
    including_answers = args.including_answers

    set_seed(args.seed)
    list_data_dict = load_strqa_jsonl(args.data_path)
    model_name_input = os.path.basename(model_name.rstrip('/'))
    
    llm = SLED_DecodedLLM(model_name, device, num_gpus, args.max_gpu_memory)
    stop_word_list = ["Q:", "\n\n##"]
    llm.set_stop_words(stop_word_list)

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


    answers = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    retry_times = args.retry

    #target = [27, 28, 31, 36, 37, 44, 45, 47, 51, 52, 62, 64, 104, 114, 131, 133, 147, 153, 168, 170, 171, 174, 177, 189, 206, 210, 227, 243, 256, 260, 276, 286, 288, 318, 323, 344, 355, 364, 366, 377, 380, 398, 410, 416, 424, 427, 452, 453, 458, 461, 470, 471, 481, 486, 499, 509, 524, 533, 534, 538, 581, 591, 616, 617, 620, 624, 636, 647, 654, 664, 665, 670, 672, 675, 679, 682, 684, 703, 707, 721, 768, 769, 772, 775, 814, 815, 852, 854, 868, 873, 879, 884, 891, 902, 911, 912, 917, 924, 931, 966, 969, 997, 999, 1013, 1042, 1051, 1054, 1066, 1086, 1091, 1096, 1104, 1111, 1116, 1158, 1159, 1165, 1167, 1185, 1195, 1225, 1230, 1243, 1244, 1254, 1265, 1266, 1276, 1279, 1284, 1293, 1317, 1337, 1338, 1345, 1348, 1350, 1356, 1359, 1367, 1376, 1377, 1385, 1398, 1421, 1425, 1429, 1440, 1452, 1457, 1467, 1500, 1514, 1524, 1530, 1533, 1535, 1538, 1559, 1560, 1580, 1584, 1587, 1596, 1600, 1604, 1618, 1647, 1673, 1687, 1689, 1702, 1727, 1731, 1734, 1735, 1762, 1795, 1800, 1804, 1814, 1825, 1841, 1842, 1847, 1848, 1854, 1863, 1898, 1915, 1922, 1925, 1929, 1946, 1951, 1955, 1982, 1994, 2010, 2011, 2014, 2022, 2025, 2032, 2052, 2072, 2100, 2113, 2115, 2141, 2145, 2166, 2176, 2186, 2187, 2188, 2189, 2195, 2214, 2232, 2248, 2250, 2274, 2281, 2286]
    #for tar in target:
    for sample in tqdm(list_data_dict):
        model_answer = None
        for i in range(retry_times):
            input_text = build_prompt(sample['question'], args.do_shuffle)
            print('inputtttttt', input_text)
            generate_kwargs = dict(single=single,ave=ave,model_name_input=model_name_input,including_answers=including_answers,th=th,ema=ema,sink_layers=sink_layers,sink=sink,beta=beta,token_weaken=token_weaken,token_enhance=token_enhance,attn_alpha=attn_alpha,start_layer=start_layer,end_layer=end_layer,max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=args.decoding_method, mature_layer=mature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top,relative_top_value=args.relative_top_value,evolution_rate=args.evolution_rate,evolution_scale=args.evolution_scale)
            model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
            for stop_word in stop_word_list:
                length_to_remove = len(stop_word)
                if model_completion[-length_to_remove:] == stop_word:
                    model_completion = model_completion[:-length_to_remove]
            model_completion = model_completion.strip()
            model_answer = clean_answer(model_completion, random_guess = (i == retry_times - 1))
            if model_answer is not None:
                break
        is_cor = is_correct(model_answer, sample['answer'])
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_answer'].append(model_answer)
        print('comppp',model_completion)
        result_dict['model_completion'].append(model_completion)
        result_dict['full_input_text'].append(input_text)



    # print(f'Num of total question: {len(answers)}, '
    #       f'correct num: {sum(answers)}, '
    #       f'correct rate: {float(sum(answers)) / len(answers)}.')

    # model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    # with open(args.output_path, 'w') as f:
    #     json.dump(result_dict, f)
