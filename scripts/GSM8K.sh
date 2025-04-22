#LLaMA 2 7B Base
python run_gsm8k.py  --model-name meta-llama/Llama-2-7b-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method VanillaGreedy
python run_gsm8k.py  --model-name meta-llama/Llama-2-7b-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method dola
python run_gsm8k.py  --model-name meta-llama/Llama-2-7b-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method SLED --evolution_rate 2 --evolution_scale 10

#LLaMA 2 7B Chat
python run_gsm8k.py  --model-name meta-llama/Llama-2-7b-chat-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method VanillaGreedy
python run_gsm8k.py  --model-name meta-llama/Llama-2-7b-chat-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method dola
python run_gsm8k.py  --model-name meta-llama/Llama-2-7b-chat-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method SLED --evolution_rate 1  --evolution_scale 5

#LLaMA 2 13B Base
python run_gsm8k.py  --model-name meta-llama/Llama-2-13b-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method VanillaGreedy
python run_gsm8k.py  --model-name meta-llama/Llama-2-13b-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method dola
python run_gsm8k.py  --model-name meta-llama/Llama-2-13b-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method SLED --evolution_rate 2 --evolution_scale 5

#LLaMA 2 13B Chat
python run_gsm8k.py  --model-name meta-llama/Llama-2-13b-chat-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method VanillaGreedy
python run_gsm8k.py  --model-name meta-llama/Llama-2-13b-chat-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method dola
python run_gsm8k.py  --model-name meta-llama/Llama-2-13b-chat-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 1 --decoding_method SLED --evolution_rate 0.25  --evolution_scale 5

#LLaMA 2 70B Base
python run_gsm8k.py  --model-name meta-llama/Llama-2-70b-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 2 --decoding_method VanillaGreedy
python run_gsm8k.py  --model-name meta-llama/Llama-2-70b-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 2 --decoding_method dola
python run_gsm8k.py  --model-name meta-llama/Llama-2-70b-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 2 --decoding_method SLED --evolution_rate 2  --evolution_scale 10

#LLaMA 2 70B Chat
python run_gsm8k.py  --model-name meta-llama/Llama-2-70b-chat-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 2 --decoding_method VanillaGreedy
python run_gsm8k.py  --model-name meta-llama/Llama-2-70b-chat-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 2 --decoding_method dola
python run_gsm8k.py  --model-name meta-llama/Llama-2-70b-chat-hf  --data-path Data/gsm8k_test --output-path output-path.json --num-gpus 2 --decoding_method SLED --evolution_rate 0.5  --evolution_scale 50
