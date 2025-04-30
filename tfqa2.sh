# (
# # 定义基础参数
# start_layer=4
# end_layer=20
# token_enhance="None"
# token_weaken="ac"
# th=0.1
# # 自动生成文件名
# output_file="/home/liang/project/concept/method/ConceptToken/tfqa_res/${start_layer}_${end_layer}.json"

# # # 执行命令（参数与文件名自动关联）
# CUDA_VISIBLE_DEVICES=4 python /home/liang/project/concept/method/ConceptToken/run_tfqa.py \
#     --model-name /home/liang/project/mmlu/llama2 \
#     --data_path /home/liang/project/mmlu/benchmark/truqa \
#     --output-path "$output_file" \
#     --num-gpus 1 \
#     --decoding_method 'attn' \
#     --start_layer "$start_layer" \
#     --end_layer "$end_layer" \
#     --attn_alpha 0.5 \
#     --token_enhance "$token_enhance" \
#     --token_weaken "$token_weaken" \
#     --beta 0 \
#     # --ema True
#     # --sink True \
#     # --sink_layers 1,2,3,4,5
# ) &

#!/bin/bash
#!/bin/bash

# 定义固定层数组合（格式："start_layer end_layer"）
layer_pairs=(
    # "4 20"
    "5 20"
    # "0 31" 
)

# 定义其他参数池
ths=(0.05)       # 阈值候选值
betas=(0)       # beta参数候选值

# 固定参数
token_enhance="None"
token_weaken="ac"
num_gpus=1
alpha=0.5

# 遍历所有参数组合
for pair in "${layer_pairs[@]}"; do
    # 解析层数组合
    IFS=' ' read -r start_layer end_layer <<< "$pair"
    
    for th in "${ths[@]}"; do
        for beta in "${betas[@]}"; do
            # 动态生成输出文件名
            output_file="/home/liang/project/concept/method/ConceptToken/tfqa_res/start${start_layer}_end${end_layer}_th${th}_beta${beta}.json"
            
            # 执行命令
            CUDA_VISIBLE_DEVICES=4 python /home/liang/project/concept/method/ConceptToken/run_tfqa.py \
                --model-name /home/liang/project/mmlu/llama2_13 \
                --data_path /home/liang/project/mmlu/benchmark/truqa \
                --output-path "$output_file" \
                --num-gpus $num_gpus \
                --decoding_method 'attn' \
                --start_layer $start_layer \
                --end_layer $end_layer \
                --attn_alpha $alpha \
                --token_enhance "$token_enhance" \
                --token_weaken "$token_weaken" \
                --th $th \
                --beta $beta  &
            
            # 控制并行进程数（按需调整）
            #wait_if_max_processes  # 需自定义进程控制函数
            wait
        done
    done
done

# 等待所有任务完成
wait

####Ori
# (
# CUDA_VISIBLE_DEVICES=0 python /home/liang/project/concept/method/sled/run_tfqa.py --model-name /home/liang/project/mmlu/llama2  --data_path /home/liang/project/concept/benchmark/truqa/data/v1 --output-path ori.json --num-gpus 1 --decoding_method VanillaGreedy > /home/liang/project/concept/method/sled/ori.txt
# ) &

# (
# CUDA_VISIBLE_DEVICES=1 python /home/liang/project/concept/method/ConceptToken/run_tfqa.py \
#     --model-name /home/liang/project/mmlu/llama2 \
#     --data_path /home/liang/project/mmlu/benchmark/truqa \
#     --output-path /home/liang/project/concept/method/ConceptToken/tfqa_res/ori.json \
#     --num-gpus 1 \
#     --decoding_method VanillaGreedy \
#     --post_softmax
# ) &

# (
# CUDA_VISIBLE_DEVICES=7 python /home/liang/project/concept/method/ConceptToken/run_tfqa.py \
#     --model-name /home/liang/project/mmlu/llama3 \
#     --data_path /home/liang/project/mmlu/benchmark/truqa \
#     --output-path /home/liang/project/concept/method/ConceptToken/tfqa_res/ori.json \
#     --num-gpus 1 \
#     --decoding_method VanillaGreedy \
# ) &

# (
# CUDA_VISIBLE_DEVICES=6,7 python /home/liang/project/concept/method/ConceptToken/run_tfqa.py \
#     --model-name /home/liang/project/mmlu/llama3 \
#     --data_path /home/liang/project/mmlu/benchmark/truqa \
#     --output-path /home/liang/project/concept/method/ConceptToken/tfqa_res/dola.json \
#     --num-gpus 2 \
#     --decoding_method dola
# ) &
# (
# CUDA_VISIBLE_DEVICES=5,4 python /home/liang/project/concept/method/ConceptToken/run_tfqa.py \
#     --model-name /home/liang/project/mmlu/llama3 \
#     --data_path /home/liang/project/mmlu/benchmark/truqa \
#     --output-path /home/liang/project/concept/method/ConceptToken/tfqa_res/sled.json \
#     --num-gpus 2 \
#     --decoding_method SLED \
#     --evolution_rate 2.5 \
#     --evolution_scale 100
# ) &

# (
# CUDA_VISIBLE_DEVICES=5 python /home/liang/project/concept/method/ConceptToken/run_tfqa.py \
#     --model-name /home/liang/project/mmlu/llama2 \
#     --data_path /home/liang/project/mmlu/benchmark/truqa \
#     --output-path /home/liang/project/concept/method/ConceptToken/tfqa_res/sled.json \
#     --num-gpus 1 \
#     --decoding_method SLED \
#     --evolution_rate 2.5 \
#     --evolution_scale 75 \
#     --post_softmax
# ) &

wait
