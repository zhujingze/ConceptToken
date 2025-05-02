output_dir="/home/liang/project/concept/method/ConceptToken/mmlu_res"
(
model_name="/home/liang/project/mmlu/llama2"
decoding_method="attn"
start_layer=4
end_layer=16
token_enhance="None"
token_weaken="ac"
sink=False
output_file="${output_dir}/${model_name##*/}_${decoding_method}_${start_layer}_${end_layer}_s${sink}.txt"
# # 执行命令（参数与文件名自动关联）
CUDA_VISIBLE_DEVICES=4,5 python /home/liang/project/concept/method/ConceptToken/run_mmlu.py \
    --model-name "${model_name}" \
    --data_path /home/liang/project/concept/method/ConceptToken/mmlu/data \
    --output-path "$output_file" \
    --num-gpus 2 \
    --decoding_method "${decoding_method}" \
    --subject 'all' \
    --start_layer "${start_layer}" \
    --end_layer "${end_layer}" \
    --attn_alpha 1.5 \
    --token_enhance "${token_enhance}" \
    --token_weaken "${token_weaken}" \
    --th 0.05 \
    --beta 1 > "${output_file}"
    #--sink "${sink}" 
) &

(
model_name="/home/liang/project/mmlu/llama2"
decoding_method="attn"
start_layer=4
end_layer=20
token_enhance="None"
token_weaken="ac"
sink=False
output_file="${output_dir}/${model_name##*/}_${decoding_method}_${start_layer}_${end_layer}_s${sink}.txt"
# # 执行命令（参数与文件名自动关联）
CUDA_VISIBLE_DEVICES=6,7 python /home/liang/project/concept/method/ConceptToken/run_mmlu.py \
    --model-name "${model_name}" \
    --data_path /home/liang/project/concept/method/ConceptToken/mmlu/data \
    --output-path "$output_file" \
    --num-gpus 2 \
    --decoding_method "${decoding_method}" \
    --subject 'all' \
    --start_layer "${start_layer}" \
    --end_layer "${end_layer}" \
    --attn_alpha 1.5 \
    --token_enhance "${token_enhance}" \
    --token_weaken "${token_weaken}" \
    --th 0.05 \
    --beta 1 > "${output_file}"
    #--sink "${sink}" 
) &
# (
# model_name="/home/liang/project/mmlu/llama2"
# decoding_method="VanillaGreedy"
# output_file="${output_dir}/${model_name##*/}_ori.txt"
# # # 执行命令（参数与文件名自动关联）
# CUDA_VISIBLE_DEVICES=4 python /home/liang/project/concept/method/ConceptToken/run_mmlu.py \
#     --model-name "${model_name}" \
#     --data_path /home/liang/project/concept/method/ConceptToken/mmlu/data \
#     --output-path "$output_file" \
#     --num-gpus 1 \
#     --decoding_method "${decoding_method}" \
#     --subject 'all' > "${output_file}"
#     #--sink "${sink}" 
# ) &


wait
