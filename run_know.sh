(
output_file="/root/zhujingze/Concept/triqa_res3/ori.json"
output_path="/root/zhujingze/Concept/triqa_res3"
output_t="/root/zhujingze/Concept/triqa_res3/ori.txt"
# 执行命令（参数与文件名自动关联）
CUDA_VISIBLE_DEVICES=0 python /root/zhujingze/Concept/run_know.py \
    --model_name /mnt/nanjgrowth-train-baidu-4-bos/zhujingze/model/llama3 \
    --output_path "$output_path" \
    --output_file "$output_file" \
    --num_gpus 1 \
    --do-rating \
    --dataset_name triviaqa \
    --decoding_method 'VanillaGreedy' > "$output_t"
) &

(
output_file="/root/zhujingze/Concept/triqa_res3/dola.json"
output_t="/root/zhujingze/Concept/triqa_res3/dola.txt"
output_path="/root/zhujingze/Concept/triqa_res3"
# 执行命令（参数与文件名自动关联）
CUDA_VISIBLE_DEVICES=1 python /root/zhujingze/Concept/run_know.py \
    --model_name /mnt/nanjgrowth-train-baidu-4-bos/zhujingze/model/llama3 \
    --output_path "$output_path" \
    --output_file "$output_file" \
    --num_gpus 1 \
    --do-rating \
    --dataset_name triviaqa \
    --decoding_method 'dola' > "$output_t"
) &

(
output_file="/root/zhujingze/Concept/triqa_res3/sled.json"
output_t="/root/zhujingze/Concept/triqa_res3/sled.txt"
output_path="/root/zhujingze/Concept/triqa_res3"
# 执行命令（参数与文件名自动关联）
CUDA_VISIBLE_DEVICES=1 python /root/zhujingze/Concept/run_know.py \
    --model_name /mnt/nanjgrowth-train-baidu-4-bos/zhujingze/model/llama3 \
    --output_path "$output_path" \
    --output_file "$output_file" \
    --num_gpus 1 \
    --do-rating \
    --dataset_name triviaqa \
    --decoding_method 'SLED' > "$output_t"
) &

(
start_layer=5
end_layer=16
token_enhance="None"
token_weaken="ac"
th=0.05
alpha=0
output_file="/root/zhujingze/Concept/triqa_res3/csa.json"
output_t="/root/zhujingze/Concept/triqa_res3/csa.txt"
output_path="/root/zhujingze/Concept/triqa_res3"
# 执行命令（参数与文件名自动关联）
CUDA_VISIBLE_DEVICES=4 python /root/zhujingze/Concept/run_know.py \
    --model_name /mnt/nanjgrowth-train-baidu-4-bos/zhujingze/model/llama3 \
    --output_path "$output_path" \
    --output_file "$output_file" \
    --num_gpus 1 \
    --do-rating \
    --dataset_name triviaqa \
    --decoding_method 'attn' \
     --start_layer $start_layer \
    --end_layer $end_layer \
    --attn_alpha $alpha \
    --token_enhance "$token_enhance" \
    --token_weaken "$token_weaken" \
    --th 0.05 \
    --ave True \
    --sink True \
    --sink_layers 0,1,2,3,4 \
    --beta 1 > $output_t
) &
