## Instructions to install python and git, install python libraries, download models and run with ONNX-runtime

### Install python
- download and install python: https://www.python.org/ftp/python/3.12.5/python-3.12.5-amd64.exe
- select checkbox 'add python to path'

### Install Git
- download and install git https://github.com/git-for-windows/git/releases/download/v2.46.0.windows.1/Git-2.46.0-64-bit.exe

### install python libraries
- `pip install -r requirements.txt`

### download ONNX models
- phi-3-mini-4k cpu onnx model:
    - https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
    - `huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*    --local-dir .`
- phi-3-mini-128k cpu onnx model
    - https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx
    - `huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .`

- microsoft/Phi-3-medium-4k-instruct-onnx-cpu
    - https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-cpu/tree/main/cpu-int4-rtn-block-32-acc-level-4

    - `huggingface-cli download microsoft/Phi-3-medium-4k-instruct-onnx-cpu --include cpu-int4-rtn-block-32-acc-level-4/* --local-dir .`

- microsoft/mistral-7b-instruct-v0.2-ONNX
    - https://huggingface.co/microsoft/mistral-7b-instruct-v0.2-ONNX
    - `huggingface-cli download microsoft/mistral-7b-instruct-v0.2-ONNX --include cpu_and_mobile/mistral-7b-instruct-v0.2-cpu-int4-rtn-block-32-acc-level-4/* --local-dir .`

### invoke models on swagger
    `uvicorn app:app --reload --host 127.0.0.1 --port 8080`
    OR
    `python -m uvicorn app:app --reload`

### run the model locally
- `python phi3-qa.py -m models/phi-3-mini-4k-instruct-onnx -v -g -l 4000`
- `python phi3-qa.py -m models/phi-3-mini-128k-instruct-onnx -v -g -l 4000`

### build an onnx model
- `python -m onnxruntime_genai.models.builder -m models\Phi-3-mini-4k-instruct -o onnx-models -p int4 -e cpu --extra_options int4_block_size=32 --extra_options int4_accuracy_level=4`

### build onnx model for gguf models
-  `python -m onnxruntime_genai.models.builder -m MistralForCausalLM -i hf-models/OpenHermes-2.5-Mistral-7B-GGUF/openhermes-2.5-mistral-7b.Q4_K_M.gguf -o onnx-built-models -p int4 -e cpu --extra_options int4_block_size=32 --extra_options int4_accuracy_level=4`

### phi3-chat-template
- chat_template = '<|system|>\n{rules}<|end>\n<|user|>\n{input}<|end|>\n   <|assistant|>'


### to select venv on jupyter notebook:
- python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

### to run mmlu model evaluation:
- python mmlu_evaluate.py --data_dir data # update model path in evaluate.py file
