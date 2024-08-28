## Update Notes
### 2024/08/15
1. InternVL2-4B model supports using openvino to accelerate the inference process. Currently only verified on Linux system and only tested on CPU platform.
### 2024/08/22
1. Support for integrated graphics on MTL, support Windows system.
### 2024/08/28
1. Add slice to llm model to optimize calculation
2. Add int8 quantization to vision model

## Running Guide
### Installation


```bash
git clone https://github.com/zhaohb/InternVL2-4B-OV.git
pip install openvino_dev 
pip install nncf
pip install transformers==4.37.2
pip install torch
pip install torchvision

Additional Operations
1. download InternVL2-4B model
2. Replace the modeling_phi3.py in the official model directory with the modeling_phi3.py in this project.
3. Delete the CUDA API under the model file.
```
### Convert InternVL2 model to OpenVINO™ IR(Intermediate Representation) and testing:
```shell
cd InternVL2-4B-OV
#for MTL iGPU windows
python.exe .\test_ov_internvl2.py -m /path/to/internvl2 -ov ..\internvl2_ov\ -llm_int4 -vision_int8 -d GPU.0

#output
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
OpenVINO version 
 2024.4.0-16419-54f58b86be2


User: Hello, who are you?
Assistant: I am an AI assistant whose name is InternVL, developed jointly by Shanghai AI Lab and SenseTime.

User: <image>
Please describe the image shortly.
Assistant: The image features a close-up view of a red panda resting on a wooden platform. The panda is characterized by its distinctive red fur, white face, and ears. The background shows a natural setting with green foliage and a wooden structure.
User: <image>
Please describe the image shortly.
Assistant: The image features a close-up view of a red panda resting on a wooden platform. The panda is characterized by its distinctive red fur, white face, and ears. The background shows a natural setting with green foliage and a wooden structure.

Vision Pre latency: 226.53 ms, Vision encoder latency: 4881.63 ms, Vision Post latency: 45.78 ms, Vision Mlp latency: 42.10 ms
LLM Model First token latency: 8775.91 ms, Output len: 55, Avage token latency: 84.71 ms
```
### Note:
After the command is executed, the IR of OpenVINO will be saved in the directory /path/to/internvl2_ov. If the existence of /path/to/internvl2_ov is detected, the model conversion process will be skipped and the IR of OpenVINO will be loaded directly.

The model: [Model link](https://hf-mirror.com/OpenGVLab/InternVL2-4B/tree/main)
### Parsing test_ov_internvl2.py's arguments :
```shell
usage: Export InternVL2 Model to IR [-h] [-m MODEL_ID] -ov OV_IR_DIR [-d DEVICE] [-pic PICTURE] [-p PROMPT] [-max MAX_NEW_TOKENS] [-llm_int4] [-vision_int8]

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
  -llm_int4, --int4_compress
                        llm int4 weights compress
  -vision_int8, --int8_quant
                        vision int8 weights qiamtize
```
