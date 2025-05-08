#subjects=("world_religions" "philosophy" "miscellaneous")
subjects=("high_school_european_history")
#subjects=("world_religions" "prehistory" "philosophy" "miscellaneous")
# ans针对attn 是否考虑<s> 特殊下沉现象 是否包含所有单词第一个token的结果 attn分布情况在不同词上情况
# wo_option针对整个task的平均结果 layer_acc & layer_consistency
# opt针对多个选项结果之间的对比 logits token还是word
#subjects=("world_religions" "college_biology" "prehistory" "high_school_biology" "philosophy")

# for subject in "${subjects[@]}"; do
#     CUDA_VISIBLE_DEVICES=6 python /data1/zjz/mmlu/test.py \
#         --model '/data1/share/model_weight/llama/llama2/7b' \
#         --device 'cuda' \
#         --data_folder '/data1/zjz/mmlu/data' \
#         --subject "${subject}" \
#         --bs 1 \
#         --method 'lens_opt' \
#         --logits_change \
#         --output_layer_acc \
#         --attn_manipulation \
#         --start_layer 0 \
#         --end_layer 2 \
#         --alpha 1.5 \
#         --token_enhance ac \
#         --token_weaken None \
#         --attn_concept \
#         # --cd
#         # --cd_change
# done

for subject in "${subjects[@]}"; do
    CUDA_VISIBLE_DEVICES=7 python /data1/zjz/mmlu/test.py \
        --model '/data1/share/model_weight/llama/llama2/7b' \
        --device 'cuda' \
        --data_folder '/data1/zjz/mmlu/data' \
        --subject "${subject}" \
        --bs 1 \
        --method 'lens_opt' \
        --logits_change \
        --output_layer_acc \
        --attn_manipulation \
        --start_layer 3 \
        --end_layer 9 \
        --alpha 1.5 \
        --token_enhance None \
        --token_weaken ac \
        --attn_concept \
        --cd
        # --cd_change
done

# (
# # for subject in "${subjects[@]}"; do
# #     CUDA_VISIBLE_DEVICES=7 python /data1/zjz/mmlu/test.py \
# #         --model '/data1/share/model_weight/llama/llama2/7b' \
# #         --device 'cuda' \
# #         --data_folder '/data1/zjz/mmlu/data' \
# #         --subject "${subject}" \
# #         --bs 1 \
# #         --method 'lens_opt' \
# #         --logits_change \
# #         --output_layer_acc \
# #         --attn_manipulation \
# #         --start_layer 3 \
# #         --end_layer 9 \
# #         --alpha 1.5 \
# #         --token_enhance ac \
# #         --token_weaken None \
# #         --attn_concept \
# #         # --cd
# #         # --cd_change
# # done

# for subject in "${subjects[@]}"; do
#     CUDA_VISIBLE_DEVICES=7 python /data1/zjz/mmlu/test.py \
#         --model '/data1/share/model_weight/llama/llama2/7b' \
#         --device 'cuda' \
#         --data_folder '/data1/zjz/mmlu/data' \
#         --subject "${subject}" \
#         --bs 1 \
#         --method 'lens_opt' \
#         --logits_change \
#         --output_layer_acc \
#         --attn_manipulation \
#         --start_layer 20 \
#         --end_layer 31 \
#         --alpha 1.5 \
#         --token_enhance ac \
#         --token_weaken None \
#         --attn_concept \
#         # --cd
#         # --cd_change
# done
# ) &

# wait
