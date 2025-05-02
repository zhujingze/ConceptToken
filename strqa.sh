# (
# # 定义基础参数
# start_layer=4
# end_layer=20
# token_enhance="None"
# token_weaken="ac"

# # 自动生成文件名
# #output_file="/home/liang/project/concept/method/ConceptToken/factor_res/cd_${start_layer}_${end_layer}_en_${token_enhance}_w_${token_weaken}.json"

# # 执行命令（参数与文件名自动关联）
# CUDA_VISIBLE_DEVICES=6 python /home/liang/project/concept/method/ConceptToken/run_strqa.py \
#     --model_name /home/liang/project/mmlu/llama2 \
#     --data_path /home/liang/project/mmlu/benchmark/strqa \
#     --output-path /home/liang/project/concept/method/ConceptToken/strqa_res \
#     --num_gpus 1 \
#     --decoding_method 'attn' \
#     --start_layer "$start_layer" \
#     --end_layer "$end_layer" \
#     --attn_alpha 0 \
#     --token_enhance "$token_enhance" \
#     --token_weaken "$token_weaken" \
#     --beta 1  \
#     # --sink True \
#     # --sink_layers 2,3
# ) &

(
# 执行命令（参数与文件名自动关联）
CUDA_VISIBLE_DEVICES=6 python /home/liang/project/concept/method/ConceptToken/run_strqa.py \
    --model_name /home/liang/project/mmlu/llama3 \
    --data_path /home/liang/project/mmlu/benchmark/strqa \
    --output_path /home/liang/project/concept/method/ConceptToken/strqa_res/sled3.json \
    --num_gpus 1 \
    --decoding_method 'SLED' \
    --evolution_rate 1.75 \
    --evolution_scale 5
) &

wait
