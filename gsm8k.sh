(
# 定义基础参数
start_layer=4
end_layer=16
token_enhance="None"
token_weaken="ac"

# 自动生成文件名
#output_file="/home/liang/project/concept/method/ConceptToken/factor_res/cd_${start_layer}_${end_layer}_en_${token_enhance}_w_${token_weaken}.json"

# 执行命令（参数与文件名自动关联）
CUDA_VISIBLE_DEVICES=3 python /home/liang/project/concept/method/ConceptToken/run_gsm8k.py \
    --model_name /home/liang/project/mmlu/llama2 \
    --data_path /home/liang/project/mmlu/benchmark/gsm8k \
    --output_path /home/liang/project/concept/method/ConceptToken/gsm8k_res/our2_ad.json \
    --num_gpus 1 \
    --decoding_method 'attn' \
    --start_layer "$start_layer" \
    --end_layer "$end_layer" \
    --attn_alpha 0 \
    --token_enhance "$token_enhance" \
    --token_weaken "$token_weaken" \
    --beta 1  \
    --th 0.05 > /home/liang/project/concept/method/ConceptToken/gsm8k_res/our2_416_ad.txt
    # --sink True \
    # --sink_layers 2,3
) &

(
# 定义基础参数
start_layer=4
end_layer=16
token_enhance="None"
token_weaken="ac"

# 自动生成文件名
#output_file="/home/liang/project/concept/method/ConceptToken/factor_res/cd_${start_layer}_${end_layer}_en_${token_enhance}_w_${token_weaken}.json"

# 执行命令（参数与文件名自动关联）
CUDA_VISIBLE_DEVICES=2 python /home/liang/project/concept/method/ConceptToken/run_gsm8k.py \
    --model_name /home/liang/project/mmlu/llama2 \
    --data_path /home/liang/project/mmlu/benchmark/gsm8k \
    --output_path /home/liang/project/concept/method/ConceptToken/gsm8k_res/our2_ad_ia.json \
    --num_gpus 1 \
    --decoding_method 'attn' \
    --start_layer "$start_layer" \
    --end_layer "$end_layer" \
    --attn_alpha 0 \
    --token_enhance "$token_enhance" \
    --token_weaken "$token_weaken" \
    --beta 1  \
    --th 0.05 \
    --including_answers True > /home/liang/project/concept/method/ConceptToken/gsm8k_res/our2_4-16_ad_ia.txt
    # --sink True \
    # --sink_layers 2,3
) &


(
# 定义基础参数
start_layer=4
end_layer=20
token_enhance="None"
token_weaken="ac"

# 自动生成文件名
#output_file="/home/liang/project/concept/method/ConceptToken/factor_res/cd_${start_layer}_${end_layer}_en_${token_enhance}_w_${token_weaken}.json"

# 执行命令（参数与文件名自动关联）
CUDA_VISIBLE_DEVICES=1 python /home/liang/project/concept/method/ConceptToken/run_gsm8k.py \
    --model_name /home/liang/project/mmlu/llama2 \
    --data_path /home/liang/project/mmlu/benchmark/gsm8k \
    --output_path /home/liang/project/concept/method/ConceptToken/gsm8k_res/our2_ad.json \
    --num_gpus 1 \
    --decoding_method 'attn' \
    --start_layer "$start_layer" \
    --end_layer "$end_layer" \
    --attn_alpha 0 \
    --token_enhance "$token_enhance" \
    --token_weaken "$token_weaken" \
    --beta 1  \
    --th 0.05 > /home/liang/project/concept/method/ConceptToken/gsm8k_res/our2_4-20_ad.txt
    # --sink True \
    # --sink_layers 2,3
) &

(
# 定义基础参数
start_layer=4
end_layer=20
token_enhance="None"
token_weaken="ac"

# 自动生成文件名
#output_file="/home/liang/project/concept/method/ConceptToken/factor_res/cd_${start_layer}_${end_layer}_en_${token_enhance}_w_${token_weaken}.json"

# 执行命令（参数与文件名自动关联）
CUDA_VISIBLE_DEVICES=0 python /home/liang/project/concept/method/ConceptToken/run_gsm8k.py \
    --model_name /home/liang/project/mmlu/llama2 \
    --data_path /home/liang/project/mmlu/benchmark/gsm8k \
    --output_path /home/liang/project/concept/method/ConceptToken/gsm8k_res/our2_ad_ia.json \
    --num_gpus 1 \
    --decoding_method 'attn' \
    --start_layer "$start_layer" \
    --end_layer "$end_layer" \
    --attn_alpha 0 \
    --token_enhance "$token_enhance" \
    --token_weaken "$token_weaken" \
    --beta 1  \
    --th 0.05 \
    --including_answers True > /home/liang/project/concept/method/ConceptToken/gsm8k_res/our2_4-20_ad_ia.txt
    # --sink True \
    # --sink_layers 2,3
) &


wait
