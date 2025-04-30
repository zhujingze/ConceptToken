# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/voidism/DoLa
import transformers
from tqdm import tqdm, trange
import argparse
from utils.utils_factor import *
from sled_decoding import SLED_DecodedLLM_Factor as SLED_DecodedLLM
import json
import warnings

transformers.logging.set_verbosity(40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num_gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data_path", type=str, default="Data/FACTOR/wiki_factor.csv")
    parser.add_argument("--output_path", type=str, default="./gsm8k_result")
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
    sink_layers=args.sink_layers
    th = args.th
    ema = args.ema
    
    list_data_dict = load_factor_dataset(args.data_path)

    
    llm = SLED_DecodedLLM(model_name, device, num_gpus, args.max_gpu_memory)
    llm.set_stop_words(["Q:", "\end{code}"])

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




    answers = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}



    for sample in tqdm(list_data_dict):
        context = sample['prefix']
        answer_true = ' ' + sample['completion']
        answers_false = []
        # print('prefix',context)
        for i in range(3):
            answers_false.append(' ' + sample[f'contradiction_{i}'])
        generate_kwargs = dict(do_sample=args.do_sample, mode=args.decoding_method, mature_layer=mature_layer, candidate_premature_layers=candidate_premature_layers,post_softmax=True, relative_top=args.relative_top, relative_top_value=args.relative_top_value,evolution_rate=args.evolution_rate,evolution_scale=args.evolution_scale)
        answer_true_log_prob, c_dist = llm.lm_score(context, answer_true, start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta, sink=sink, sink_layers=sink_layers,th=th,ema=ema,**generate_kwargs)
        # print('true',answer_true)
        answer_false_log_probs = []
        for answer_false in answers_false:
            # print('false',answer_false)
            answer_false_log_prob, c_dist = llm.lm_score(context, answer_false, start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta,sink=sink,sink_layers=sink_layers,th=th,ema=ema, **generate_kwargs)

            answer_false_log_probs.append(answer_false_log_prob)

        is_cor = True

        for answer_false_log_prob in answer_false_log_probs:
            if answer_true_log_prob < answer_false_log_prob:
                is_cor = False
                break

        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_completion'].append([answer_true_log_prob] + answer_false_log_probs)


    # print(f'{args.start_layer}-{args.end_layer}_{args.attn_alpha}, Num of total question: {len(answers)}, '
    #     f'correct num: {sum(answers)}, '
    #     f'correct rate: {float(sum(answers))/len(answers)}.')
    if 'wiki' in args.data_path:
        sub = 'wiki'
    elif 'news' in args.data_path:
        sub = 'news'
    elif 'expert' in args.data_path:
        sub = 'expert'
        
    if 'llama3' in model_name:
        namemodel = 'llama3'
    else:
        namemodel = 'llama2'
    print(f'{namemodel}_{args.decoding_method}_{sub}_{args.start_layer}-{args.end_layer}_{args.attn_alpha}_{args.beta}_s{args.sink}, correct rate: {float(sum(answers))/len(answers)}.')

    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    with open(args.output_path, 'w') as f:
        json.dump(result_dict, f)
