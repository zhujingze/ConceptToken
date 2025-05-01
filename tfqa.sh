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

wait
