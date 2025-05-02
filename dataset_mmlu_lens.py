import pandas as pd
from tqdm import tqdm

def prepare_mc_dataset(subject, csv_file, example_file, tokenizer, max_length=2048):
    # 加载数据集
    data_df = pd.read_csv(csv_file, header=None)
    example_df = pd.read_csv(example_file, header=None)
    choices = ["A", "B", "C", "D"]
    dataset = []

    # 预处理示例数据
    example_cache = {}
    for k in range(min(5, len(example_df))+1):
        example_text = ""
        for idx in range(k):
            ex_question = example_df.iloc[idx, 0]
            ex_choices = [example_df.iloc[idx, i+1] for i in range(4)]
            ex_answer = example_df.iloc[idx, 5]
            correct_idx = choices.index(ex_answer)
            example_text += f"Question: {ex_question}\nAnswer: {ex_choices[correct_idx]}\n\n"
        example_cache[k] = example_text

    # 处理每个问题
    #for data_row in tqdm(data_df.itertuples(index=False), total=len(data_df)):
    for data_row in data_df.itertuples(index=False):
        question = data_row[0]
        current_choices = [data_row[i+1] for i in range(4)]
        answer = data_row[5]
        
        # 动态寻找最优k值
        best_k = 0
        best_prefix = ""
        for k in range(min(5, len(example_df)), -1, -1):
            base_prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n{example_cache[k]}"
            input_text = f"Question: {question}\nAnswer:"
            candidate_prefix = base_prompt + input_text
            
            # 检查所有选项长度
            valid = True
            for choice in current_choices:
                full_text = candidate_prefix + " " + str(choice)
                inputs = tokenizer(full_text, return_tensors="pt", truncation=False)
                if inputs.input_ids.shape[-1] > max_length:
                    valid = False
                    break
            
            if valid:
                best_k = k
                best_prefix = candidate_prefix
                break

        # 构建答案选项
        label = choices.index(answer)
        true_answer = " " + str(current_choices[label])
        false_answers = [" " + str(c) for i, c in enumerate(current_choices) if i != label]

        dataset.append({
            "prefix": best_prefix,
            "true_answer": true_answer,
            "false_answers": false_answers,
            "subject": subject
        })

    return dataset
