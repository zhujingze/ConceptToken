import transformers
from tqdm import tqdm, trange
import argparse
from utils.utils_gsm8k import *
from sled_decoding_gen import SLED_DecodedLLM_GSM8K as SLED_DecodedLLM
import json
import warnings
transformers.logging.set_verbosity(40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num_gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data_path", type=str, default="./gsm8k")
    parser.add_argument("--output_path", type=str, default="./gsm8k_result")
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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--decoding_method", type=str, default="VanillaGreedy", choices=["VanillaGreedy","SLED", "dola","attn"])
    parser.add_argument("--evolution_rate", type=float, default=2)
    parser.add_argument("--evolution_scale", type=int, default=10)
    parser.add_argument("--start_layer", type=int)
    parser.add_argument("--end_layer", type=int)
    parser.add_argument("--attn_alpha", type=float)
    parser.add_argument("--token_enhance", type=str)
    parser.add_argument("--token_weaken", type=str)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--th", type=float)
    parser.add_argument("--ema", type=bool)
    parser.add_argument("--single", type=bool)
    parser.add_argument("--ave", type=bool)
    parser.add_argument("--including_answers", type=bool)
    parser.add_argument("--sink", type=bool)
    parser.add_argument("--sink_layers",
                   type=lambda s: [int(x) for x in s.split(',')],
                   default=[],
                   help="要启用的层号列表，用逗号分隔（例如：'1,3,5'）")


    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    set_seed(args.seed)
    list_data_dict = load_gsm8k_jsonl(args.data_path)
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
    stop_word_list = ["Q:", "\end{code}"]
    llm.set_stop_words(stop_word_list)

    if args.decoding_method in ["VanillaGreedy","attn"]:
        if args.early_exit_layers is not None:
            warnings.warn("The 'early_exit_layers' argument should be None when using Vanilla greedy decoding.")
        print("Vanilla greedy decoding from the final layer", flush=True)
        mature_layer = None
        candidate_premature_layers = None

    else:
        if args.early_exit_layers is None:
            early_exit_layers = [int(x) for x in range(llm.num_layers+1)]
        else:
            early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]

        print(f"MODE: {args.decoding_method} decoding with the final layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mature_layer = early_exit_layers[-1]
        candidate_premature_layers = early_exit_layers[:-1]



    answers = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    for sample in tqdm(list_data_dict[:]):
        input_text = build_prompt(sample['instruction'], args.do_shuffle)
        generate_kwargs = dict(single=single,ave=ave,max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p,
                               including_answers=including_answers,th=th,ema=ema,model_name_input=model_name_input,
                               sink_layers=sink_layers,sink=sink,beta=beta,token_weaken=token_weaken,token_enhance=token_enhance,
                               attn_alpha=attn_alpha,start_layer=start_layer,end_layer=end_layer,
                               top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty,
                               mode=args.decoding_method, mature_layer=mature_layer,
                               candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top,
                               relative_top_value=args.relative_top_value,evolution_rate=args.evolution_rate,evolution_scale=args.evolution_scale)
        model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, sample['output'])
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_answer'].append(model_answer)
        result_dict['model_completion'].append(model_completion)
        result_dict['full_input_text'].append(input_text)


    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    with open(args.output_path , 'w') as f:
        json.dump(result_dict, f)

    print(f'Num of total question: {len(answers)}, '
         f'correct num: {sum(answers)}, '
         f'correct rate: {float(sum(answers)) / len(answers)}.')
