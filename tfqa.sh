(
CUDA_VISIBLE_DEVICES=0 python /home/liang/project/concept/method/sled/run_tfqa.py \
    --model-name /home/liang/project/mmlu/llama2 \
    --data_path /home/liang/project/concept/benchmark/truqa/data/v0 \
    --output-path 2-16_c.json \
    --num-gpus 1 \
    --decoding_method 'attn' \
    --start_layer 2 \
    --end_layer 16 \
    --attn_alpha 1.5 \
    --token_enhance None \
    --token_weaken ac \
    --alpha 1
) &

(
CUDA_VISIBLE_DEVICES=2 python /home/liang/project/concept/method/sled/run_tfqa.py \
    --model-name /home/liang/project/mmlu/llama2 \
    --data_path /home/liang/project/concept/benchmark/truqa/data/v0 \
    --output-path 2-9_p.json \
    --num-gpus 1 \
    --decoding_method 'attn' \
    --start_layer 2 \
    --end_layer 9 \
    --attn_alpha 1.5 \
    --token_enhance None \
    --token_weaken p \
    --alpha 1
) &

(
CUDA_VISIBLE_DEVICES=3 python /home/liang/project/concept/method/sled/run_tfqa.py \
    --model-name /home/liang/project/mmlu/llama2 \
    --data_path /home/liang/project/concept/benchmark/truqa/data/v0 \
    --output-path 2-9_c.json \
    --num-gpus 1 \
    --decoding_method 'attn' \
    --start_layer 2 \
    --end_layer 9 \
    --attn_alpha 1.5 \
    --token_enhance None \
    --token_weaken ac \
    --alpha 1
) &

(
CUDA_VISIBLE_DEVICES=4 python /home/liang/project/concept/method/sled/run_tfqa.py \
    --model-name /home/liang/project/mmlu/llama2 \
    --data_path /home/liang/project/concept/benchmark/truqa/data/v0 \
    --output-path 8-16_c.json \
    --num-gpus 1 \
    --decoding_method 'attn' \
    --start_layer 8 \
    --end_layer 16 \
    --attn_alpha 1.5 \
    --token_enhance None \
    --token_weaken ac \
    --alpha 1
) &
# (
# CUDA_VISIBLE_DEVICES=0 python /home/liang/project/concept/method/sled/run_tfqa.py --model-name /home/liang/project/mmlu/llama2  --data_path /home/liang/project/concept/benchmark/truqa/data/v1 --output-path ori.json --num-gpus 1 --decoding_method VanillaGreedy > /home/liang/project/concept/method/sled/ori.txt
# ) &
# (
# CUDA_VISIBLE_DEVICES=4 python /home/liang/project/concept/method/sled/run_tfqa.py --model-name /home/liang/project/mmlu/llama2  --data_path /home/liang/project/concept/benchmark/truqa/data/v1 --output-path dola.json --num-gpus 1 --decoding_method dola > /home/liang/project/concept/method/sled/dola.txt
# ) &
# (
# CUDA_VISIBLE_DEVICES=5 python /home/liang/project/concept/method/sled/run_tfqa.py --model-name /home/liang/project/mmlu/llama2  --data_path /home/liang/project/concept/benchmark/truqa/data/v1 --output-path sled.json --num-gpus 1 --decoding_method SLED --evolution_rate 2.5  --evolution_scale 75 > /home/liang/project/concept/method/sled/sled.txt
# ) &

wait