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
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="Data/FACTOR/wiki_factor.csv")
    parser.add_argument("--output-path", type=str, default="./gsm8k_result")
    parser.add_argument("--early-exit-layers", type=str, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoding_method", type=str, default="VanillaGreedy", choices=["VanillaGreedy","SLED", "dola"])
    parser.add_argument("--evolution_rate", type=float, default=2)
    parser.add_argument("--evolution_scale", type=int, default=10)

    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    list_data_dict = load_factor_dataset(args.data_path)

    
    llm = SLED_DecodedLLM(model_name, device, num_gpus, args.max_gpu_memory)
    llm.set_stop_words(["Q:", "\end{code}"])

    if args.decoding_method == "VanillaGreedy":
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
        for i in range(3):
            answers_false.append(' ' + sample[f'contradiction_{i}'])
        generate_kwargs = dict(do_sample=args.do_sample, mode=args.decoding_method, mature_layer=mature_layer, candidate_premature_layers=candidate_premature_layers,post_softmax=True, relative_top=args.relative_top, relative_top_value=args.relative_top_value,evolution_rate=args.evolution_rate,evolution_scale=args.evolution_scale)
        answer_true_log_prob, c_dist = llm.lm_score(context, answer_true, **generate_kwargs)

        answer_false_log_probs = []
        for answer_false in answers_false:
            answer_false_log_prob, c_dist = llm.lm_score(context, answer_false, **generate_kwargs)

            answer_false_log_probs.append(answer_false_log_prob)

        is_cor = True

        for answer_false_log_prob in answer_false_log_probs:
            if answer_true_log_prob < answer_false_log_prob:
                is_cor = False
                break

        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_completion'].append([answer_true_log_prob] + answer_false_log_probs)


    print(f'Num of total question: {len(answers)}, '
        f'correct num: {sum(answers)}, '
        f'correct rate: {float(sum(answers))/len(answers)}.')

    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    with open(args.output_path, 'w') as f:
        json.dump(result_dict, f)
