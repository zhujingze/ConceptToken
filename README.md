We introduce <strong>LayerCake</strong>, a decoding-time framework that bridges transformer layer depth with token semantics by aligning different token types to the specific layers where they play the most critical roles in factual reasoning.

## Installation
- **Python**: Recommended to use Python 3.10 or higher.
- **PyTorch**: We recommend using PyTorch version 2.5.1 with CUDA 12.2.
- **NLTK**: We provide the NLTK tools used, and recommend copying the folder into the 'share' directory of your environment.
- **Transformers**: Install the `transformers` library from the local directory included in the project folder.
  ```bash
  pip install -e transformers
  ```
- **Other Dependencies**: 
  ```bash
  pip install -r requirements.txt
  ```

## Evaluation
Below we provide example scripts for running our method `attn` and other baseline methods such as `SLED`, `dola` and `Greedy Decoding`.

### FACTOR (Multiple Choices)
  
```bash
python run_factor.py --model_name meta-llama/Llama-2-7b-hf  --data_path Data/FACTOR/expert_factor.csv  --output_path output-path.json --num_gpus 1 --decoding_method VanillaGreedy
python run_factor.py --model_name meta-llama/Llama-2-7b-hf  --data_path Data/FACTOR/expert_factor.csv  --output_path output-path.json --num_gpus 1 --decoding_method dola
python run_factor.py --model_name meta-llama/Llama-2-7b-hf  --data_path Data/FACTOR/expert_factor.csv  --output_path output-path.json --num_gpus 1 --decoding_method SLED --evolution_rate 2  --evolution_scale 10
python run_factor.py --model_name meta-llama/Llama-2-7b-hf  --data_path Data/FACTOR/expert_factor.csv  --output_path output-path.json --num_gpus 1 --decoding_method attn --start_layer 5
--end_layer 16 --token_weaken ac --th 0.05 --sink True --sink_layers 0,1,2,3,4 --beta 1
```

### TruthfulQA (Multiple Choices)
  
```bash
python run_tfqa.py --model_name meta-llama/Llama-2-7b-hf  --data_path Data/TruthfulQA --output_path output-path.json --num_gpus 1 --decoding_method VanillaGreedy
python run_tfqa.py --model_name meta-llama/Llama-2-7b-hf  --data_path Data/TruthfulQA --output_path output-path.json --num_gpus 1 --decoding_method dola
python run_tfqa.py --model_name meta-llama/Llama-2-7b-hf  --data_path Data/TruthfulQA --output_path output-path.json --num_gpus 1 --decoding_method SLED --evolution_rate 2.5  --evolution_scale 75
python run_factor.py --model-name meta-llama/Llama-2-7b-hf  --data_path Data/TruthfulQA  --output_path output-path.json --num_gpus 1 --decoding_method attn --start_layer 5
--end_layer 16 --token_weaken ac --th 0.05 --sink True --sink_layers 0,1,2,3,4 --beta 0
```

### StrategyQA 
  
```bash
python run_strqa.py  --model_name meta-llama/Llama-2-7b-hf  --data_path Data/StrategyQA --output_path output-path.json --num_gpus 1 --decoding_method VanillaGreedy
python run_strqa.py  --model_name meta-llama/Llama-2-7b-hf  --data_path Data/StrategyQA --output_path output-path.json --num_gpus 1 --decoding_method dola
python run_strqa.py  --model_name meta-llama/Llama-2-7b-hf  --data_path Data/StrategyQA --output_path output-path.json --num_gpus 1 --decoding_method SLED --evolution_rate 1.75 --evolution_scale 5
python run_strqa.py  --model_name meta-llama/Llama-2-7b-hf  --data_path Data/StrategyQA --output_path output-path.json --num_gpus 1 --decoding_method attn --start_layer 5
--end_layer 16 --token_weaken ac --th 0.05 --ave True --sink True --sink_layers 0,1,2,3,4 --beta 1
```
### HellaSwag
```bash
python run_hellaswag.py --model_name meta-llama/Llama-2-7b-hf  --data_path Data/Hellaswag  --output_path output-path.json --num_gpus 1 --decoding_method VanillaGreedy
python run_hellaswag.py --model_name meta-llama/Llama-2-7b-hf  --data_path Data/Hellaswag  --output_path output-path.json --num_gpus 1 --decoding_method dola
python run_hellaswag.py --model_name meta-llama/Llama-2-7b-hf  --data_path Data/Hellaswag  --output_path output-path.json --num_gpus 1 --decoding_method SLED --evolution_rate 2  --evolution_scale 10
python run_hellaswag.py --model_name meta-llama/Llama-2-7b-hf  --data_path Data/Hellaswag  --output_path output-path.json --num_gpus 1 --decoding_method attn --start_layer 5
--end_layer 16 --token_weaken ac --th 0.05 --sink True --sink_layers 0,1,2,3,4 --beta 1
```

Additional experiments involving various models can be found in the `scripts` folder.














