import argparse
import openvino as ov
from pathlib import Path
from ov_internvl2 import OVInternVLForCausalLM, InternVL2_OV
from transformers import TextStreamer
import time
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser("Export InternVL2 Model to IR", add_help=True)
    parser.add_argument("-m", "--model_id", required=True, help="model_id or directory for loading")
    parser.add_argument("-o", "--output_dir", required=True, help="output directory for saving model")
    parser.add_argument('-d', '--device', default='CPU', help='inference device')
    parser.add_argument('-pic', '--picture', default="./moondream.jpg", help='picture file')
    parser.add_argument('-p', '--prompt', default="Describe this image.", help='prompt')
    parser.add_argument('-max', '--max_new_tokens', default=256, help='max_new_tokens')
    parser.add_argument('-int4', '--int4_compress', action="store_true", help='int4 weights compress')

    args = parser.parse_args()
    model_id = args.model_id
    ov_model_path = args.output_dir
    device = args.device
    max_new_tokens = args.max_new_tokens
    picture_path = args.picture
    question = args.prompt
    int4_compress = args.int4_compress

    if not Path(ov_model_path).exists():
        internvl2_ov = InternVL2_OV(pretrained_model_path=model_id, ov_model_path=ov_model_path, device=device, int4_compress=int4_compress)
        internvl2_ov.export_vision_to_ov()
        del internvl2_ov.model
        del internvl2_ov.tokenizer
        del internvl2_ov
    elif Path(ov_model_path).exists() and int4_compress is True and not Path(f"{ov_model_path}/llm_stateful_int4.xml").exists():
        internvl2_ov = InternVL2_OV(pretrained_model_path=model_id, ov_model_path=ov_model_path, device=device, int4_compress=int4_compress)
        internvl2_ov.export_vision_to_ov()
        del internvl2_ov.model
        del internvl2_ov.tokenizer
        del internvl2_ov
    
    llm_infer_list = []
    core = ov.Core()
    internvl2_model = OVInternVLForCausalLM(core=core, ov_model_path=ov_model_path, device=device, int4_compress=int4_compress, llm_infer_list=llm_infer_list)

    generation_config = {
            "bos_token_id": internvl2_model.tokenizer.bos_token_id,
            "pad_token_id": internvl2_model.tokenizer.bos_token_id,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
    question = 'Hello, who are you?'
    response, history = internvl2_model.chat(None, question, generation_config, history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

    for i in range(2):
        pixel_values = internvl2_model.load_image("../InternVL2-4B/examples/image1.jpg")

        generation_config = {
            "bos_token_id": internvl2_model.tokenizer.bos_token_id,
            "pad_token_id": internvl2_model.tokenizer.bos_token_id,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

        question = '<image>\nPlease describe the image shortly.'
        response = internvl2_model.chat(pixel_values, question, generation_config)
        print(f'User: {question}\nAssistant: {response}')

        ## i= 0 is warming up
        if i != 0:
            print("\n\n")
            if len(llm_infer_list) > 1:
                avg_token = sum(llm_infer_list[1:]) / (len(llm_infer_list) - 1)
                print(f"First token latency: {llm_infer_list[0]:.2f} ms, Output len {len(llm_infer_list) - 1}, Avage token latency: {avg_token:.2f} ms")



