## Update Notes
### 2024/08/015
1. InternVL2-4B model supports using openvino to accelerate the inference process. Currently only verified on Linux system and only tested on CPU platform.

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
#for cpu
python3 test_ov_internvl2.py -m /path/to/internvl2 -o /path/to/internvl2_ov

#output
OpenVINO version 
 2024.4.0-16352-12eae34397e

User: Hello, who are you?
Assistant: I am an AI assistant whose name is InternVL, developed jointly by Shanghai AI Lab and SenseTime.

User: <image>
Please describe the image shortly.
Assistant: The image shows a red panda resting on a wooden platform. The red panda has a distinctive reddish-brown fur, white face, and ears, and it appears to be looking directly at the camera. The background features green foliage, suggesting that the red panda is in a natural or semi-natural habitat.
User: <image>
Please describe the image shortly.
Assistant: The image shows a red panda resting on a wooden platform. The red panda has a distinctive reddish-brown fur, white face, and ears, and it appears to be looking directly at the camera. The background features green foliage, suggesting that the red panda is in a natural or semi-natural habitat.

First token latency: 19272.82 ms, Output len 71, Avage token latency: 909.52 ms
```
### Note:
After the command is executed, the IR of OpenVINO will be saved in the directory /path/to/internvl2_ov. If the existence of /path/to/internvl2_ov is detected, the model conversion process will be skipped and the IR of OpenVINO will be loaded directly.

The model: [Model link]()
### Parsing test_ov_internvl2.py's arguments :
```shell
usage: Export InternVL2 Model to IR [-h] -m MODEL_ID -o OUTPUT_DIR [-d DEVICE] [-pic PICTURE] [-p PROMPT] [-max MAX_NEW_TOKENS] [-int4]

options:
  -h, --help            show this help message and exit
  -m MODEL_ID, --model_id MODEL_ID
                        model_id or directory for loading
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
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

