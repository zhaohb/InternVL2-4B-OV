## Update Notes
### 2024/08/15
1. InternVL2-4B model supports using openvino to accelerate the inference process. Currently only verified on Linux system and only tested on CPU platform.
### 2024/08/22
1. Support for integrated graphics on MTL

## Running Guide
### Installation


```bash
git clone https://github.com/zhaohb/InternVL2-4B-OV.git
pip install openvino_dev 
pip install nncf
pip install transformers==4.37.2
pip install torch
pip install torchvision
```
### Convert InternVL2 model to OpenVINO™ IR(Intermediate Representation) and testing:
```shell
cd InternVL2-4B-OV
#for MTL iGPU windows
python.exe .\test_ov_internvl2.py -ov ..\internvl2_ov\ -int4 -d GPU.0

#output
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
OpenVINO version 
 2024.4.0-16419-54f58b86be2


User: Hello, who are you?
Assistant: I am an AI assistant whose name is InternVL, developed jointly by Shanghai AI Lab and SenseTime.

User: <image>
Please describe the image shortly.
Assistant: The image features a red panda, a species native to China, with its distinctive red fur and white face. The panda is perched on a wooden platform, with its head resting on a wooden beam. The background shows green foliage, suggesting a natural or semi-natural habitat.
User: <image>
Please describe the image shortly.
Assistant: The image features a red panda, a species native to China, with its distinctive red fur and white face. The panda is perched on a wooden platform, with its head resting on a wooden beam. The background shows green foliage, suggesting a natural or semi-natural habitat.

Vision Pre latency: 66.18 ms, Vision encoder latency: 4741.70 ms, Vision Post latency: 42.73 ms, Vision Mlp latency: 19.14 ms
LLM Model First token latency: 14256.23 ms, Output len: 62, Avage token latency: 163.01 ms
```
### Note:
After the command is executed, the IR of OpenVINO will be saved in the directory /path/to/internvl2_ov. If the existence of /path/to/internvl2_ov is detected, the model conversion process will be skipped and the IR of OpenVINO will be loaded directly.

The model: [Model link](https://hf-mirror.com/OpenGVLab/InternVL2-4B/tree/main)
### Parsing test_ov_internvl2.py's arguments :
```shell
usage: Export InternVL2 Model to IR [-h] [-m MODEL_ID] -ov OV_IR_DIR [-d DEVICE] [-pic PICTURE] [-p PROMPT] [-max MAX_NEW_TOKENS] [-int4]

options:
  -h, --help            show this help message and exit
  -m MODEL_ID, --model_id MODEL_ID
                        model_id or directory for loading
  -ov OV_IR_DIR, --ov_ir_dir OV_IR_DIR
                        output directory for saving model
  -d DEVICE, --device DEVICE
                        inference device
  -pic PICTURE, --picture PICTURE
                        picture file
  -p PROMPT, --prompt PROMPT
                        prompt
  -max MAX_NEW_TOKENS, --max_new_tokens MAX_NEW_TOKENS
                        max_new_tokens
  -int4, --int4_compress
                        int4 weights compress
```

