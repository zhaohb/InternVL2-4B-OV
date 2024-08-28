import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import AutoConfig
from typing import List
import logging as log
from pathlib import Path
from transformers.generation import GenerationConfig, GenerationMixin
import numpy as np
from openvino.runtime import opset13
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
import PIL
from PIL import Image

from typing import Optional, Tuple, List, Union

import openvino as ov
from openvino.runtime import Core, Type
from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher
from openvino.runtime import opset10 as ops
from openvino.preprocess import PrePostProcessor
import nncf

import time
import warnings

from conversation import get_conv_template

def model_has_state(ov_model: ov.Model):
    # TODO: Provide a better way based on the variables availability, but OV Python API doesn't expose required methods
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):   # TODO: Can we derive the dimensions from the model topology?
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    gather_dim: int,
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model):
    key_value_input_names = [
        key.get_any_name() for key in ov_model.inputs if any("key_values" in key_name for key_name in key.get_names())
    ]
    key_value_output_names = [
        key.get_any_name() for key in ov_model.outputs if any("present" in key_name for key_name in key.get_names())
    ]
    not_kv_inputs = [
        input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())
    ]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1
    
    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )   

class InsertSlice(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Result")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            print("root: ", root)
            if root is None:
                return False
            root_output = matcher.get_match_value()
            print("root_output", root_output)
            root_name = root.get_friendly_name()
            if (len(root.get_output_partial_shape(0)) == 3):
                print(f"Find target root node name: {root_name}")
                parent = root.input_value(0).get_node()
                print(f"Find target parent node name: {parent.get_friendly_name()}")
                grand_parent = parent.input_value(0).get_node()
                print(f"Find grandparent node name: {grand_parent.get_friendly_name()}")
                grand_parent_output = parent.input(0).get_source_output()
                print("grand_parent_output: ", grand_parent_output)
                consumers = grand_parent_output.get_target_inputs()
                
                print(f"consumers: {consumers}")
                print("Original reshape node output shape:", grand_parent_output.get_partial_shape())
                start = np.array([0, -1, 0], dtype=np.int32)
                stop = np.array([1, -2, 3072], dtype=np.int32)
                step = np.array([1, -1, 1], dtype=np.int32)
                axes = np.array([0, 1, 2], dtype=np.int32)
                slice = ops.slice(grand_parent, start, stop, step, axes, name="inserted_slice")
                print("After insert slice node, output shape:", slice.output(0).get_partial_shape())

                for consumer in consumers:
                    consumer.replace_source_output(slice.output(0))
                self.model_changed = True
                # Use new operation for additional matching
                self.register_new_node(slice)
                                
                return True

        self.register_matcher(Matcher(param,"InsertSlice"), callback)
class LlmStatefulModel():
    def __init__(
        self,
        model=None,
        tokenizer=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
        int4_compress=False,
    ):
        self.name = "InternVL2 LLM Model"
        self.model = model
        self.tokenizer = tokenizer
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.int4_compress = int4_compress
        self.inputs_dict = {}

    def get_model(self):
        return self.model.language_model

    def get_input_names(self):
        inputs = ['attention_mask', 'position_ids']
        for idx in range(len(self.model.language_model.model.layers)):
            inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
        inputs.append('inputs_embeds')
        return inputs

    def get_output_names(self):
        outputs = ['logits']
        for idx in range(len(self.model.language_model.model.layers)):
            outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])
        return outputs

    def get_dynamic_axes(self):
        pass

    def get_sample_input(self):
            pass
    
    def save_tokenizer(self, tokenizer, out_dir):
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as e:
            log.error(f'tokenizer loading failed with {e}')

    def convert_sdpa_ov(self):
        llm_model = self.get_model()        
        attention_mask = torch.ones(1, 77)

        llm_input = torch.rand(( 1, 77, 3072), dtype=torch.float32)
        pkv = llm_model(inputs_embeds=llm_input, attention_mask=attention_mask, use_cache=True, return_dict=False)[1]

        attention_mask = torch.ones(1, 77*2)
        import numpy as np
        position_ids = torch.tensor([[77*2-1]])

        llm_model.config.torchscript = True
        ov_model = ov.convert_model(
            llm_model,
            example_input={
                "inputs_embeds":  llm_input,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": pkv,
             },
        )
        # print("stateful model inputs: ", ov_model.inputs)
        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        patch_stateful(ov_model)
        manager = Manager()
        manager.register_pass(InsertSlice())
        manager.run_passes(ov_model)

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/llm_stateful.xml"))
        self.save_tokenizer(self.tokenizer, self.ov_model_path)
        self.model.config.save_pretrained(self.ov_model_path)

        if self.int4_compress:
            compression_configuration = {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 1,
            }
            ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
            ov.save_model(ov_compressed_model, Path(f"{self.ov_model_path}/llm_stateful_int4.xml"))
    
class LlmEmbdModel():
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
    ):
        self.name = "InternVL2 LLM Embd Model"
        self.model = model
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.inputs_dict = {}

    def get_model(self):
        return self.model.language_model.model.embed_tokens

    def get_input_names(self):
        inputs = ['input_ids']
        return inputs

    def get_output_names(self):
        outputs = ['inputs_embeds']
        return outputs

    def get_dynamic_axes(self):
        pass

    def get_sample_input(self):
            pass

    def convert_sdpa_ov(self):
        embd_model = self.get_model()        

        input_ids = torch.randint(0, 32020, ( 1, 3408))

        ov_model = ov.convert_model(
            embd_model,
            example_input={
                "input":  input_ids,
             },
        )

        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/llm_embd.xml"))

class VisionMlpModel():
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
    ):
        self.name = "Vision Mlp Model"
        self.model = model
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.inputs_dict = {}

    def get_model(self):
        return self.model.mlp1

    def get_input_names(self):
        return ['vit_embeds']

    def get_output_names(self):
        outputs = ['vit_mlp']
        return outputs

    def get_sample_input(self):
            pass

    def convert_sdpa_ov(self):
        encoder_model = self.get_model()      
        inputs_embeds =  torch.rand(( 13, 256, 4096), dtype=torch.float32)  
        ov_model = ov.convert_model(
            encoder_model,
            example_input={
                "input": inputs_embeds,
             },
        )
        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/vision_mlp.xml"))

import requests
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from datasets import load_dataset
import tqdm

class VisionModel():
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
        int8_quant=False,
    ):
        self.name = "Vision Encoder Model"
        self.model = model
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.inputs_dict = {}
        self.vision_pre_process = Preprocess()
        self.int8_quant = int8_quant

    def get_model(self):
        return self.model.vision_model

    def get_input_names(self):
        return ['pixel_values']

    def get_output_names(self):
        outputs = ['vision_output']
        return outputs

    def get_sample_input(self):
            pass

    def get_pil_from_url(self, url):
        """
        Downloads and converts an image from a URL to a PIL Image object.
        """
        response = requests.get(url, verify=False, timeout=20)
        image = Image.open(BytesIO(response.content))
        return image.convert("RGB")

    def collate_fn(self, example, image_column="image_url"):
        """
        Preprocesses an example by loading and transforming image and text data.
        Checks if the text data in the example is valid by calling the `check_text_data` function.
        Downloads the image specified by the URL in the image_column by calling the `get_pil_from_url` function.
        If there is any error during the download process, returns None.
        Returns the preprocessed inputs with transformed image and text data.
        """
        # print(example, image_column)
        assert len(example) == 1
        example = example[0]
        url = example[image_column]
        try:
            image = self.get_pil_from_url(url)
            h, w = image.size
            # print("h, w: ", h,w)
            if h == 1 or w == 1:
                return None
        except Exception:
            return None

        inputs = self.vision_pre_process.load_image_quant(image)
        return inputs
    
    def prepare_calibration_data(self, dataloader, init_steps):
        """
        This function prepares calibration data from a dataloader for a specified number of initialization steps.
        It iterates over the dataloader, fetching batches and storing the relevant data.
        """
        data = []
        print(f"Fetching {init_steps} samples for the initialization...")
        for batch in dataloader:
            if len(data) == init_steps:
                break
            if batch is not None:
                with torch.no_grad():
                    data.append(
                        {
                            "pixel_values": batch
                        })
        return data

    def prepare_dataset(self, opt_init_steps=50, max_train_samples=1000):
        """
        Prepares a vision-text dataset for quantization.
        """
        dataset = load_dataset("google-research-datasets/conceptual_captions", trust_remote_code=True)
        train_dataset = dataset["train"].shuffle(seed=42)
        dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=self.collate_fn, batch_size=1, pin_memory=True)
        # breakpoint()
        calibration_data = self.prepare_calibration_data(dataloader, opt_init_steps)
        return calibration_data

    def convert_sdpa_ov(self):
        vison_model = self.get_model()       
        pixel_values =  torch.rand(( 1, 3, 448, 448), dtype=torch.float32)
        ov_model = ov.convert_model(
            vison_model,
            example_input={
                "pixel_values": pixel_values,
             },
        )

        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})
        
        ## for iGPU
        shapes = {}     
        for input_layer  in ov_model.inputs:
            shapes[input_layer] = input_layer.partial_shape
            shapes[input_layer][2] = 448  #w
            shapes[input_layer][3] = 448  #h
        ov_model.reshape(shapes)
        ov.save_model(ov_model, Path(f"{self.ov_model_path}/vision.xml"))
        
        if self.int8_quant:

            calibration_data = self.prepare_dataset()
            calibration_dataset = nncf.Dataset(calibration_data)
            quantized_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=calibration_dataset,
                model_type=nncf.ModelType.TRANSFORMER,
                subset_size=len(calibration_data),
                # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
                advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.6)
            )
            ## for iGPU
            shapes = {}     
            for input_layer  in quantized_model.inputs:
                shapes[input_layer] = input_layer.partial_shape
                shapes[input_layer][0] = 1  #w
                shapes[input_layer][2] = 448  #w
                shapes[input_layer][3] = 448  #h
            quantized_model.reshape(shapes)
            ov.save_model(quantized_model, Path(f"{self.ov_model_path}/vision_int8.xml"))

class Preprocess:
    def __init__(self):
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image_quant(self, image, input_size=448, max_num=12):
        # image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    
class Postprocess:
    def __init__(self):
        self.downsample_ratio = 0.5

        vit_hidden_size = 1024
        llm_hidden_size = 3072

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        self.ps_version = "V2"
        
    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def postprocess(self, vit_embeds=None):
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        # vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

class InternVL2_OV:
    def __init__(self, pretrained_model_path=None, model=None, tokenizer=None, ov_model_path='/tmp/moonstream2_ov/', device='CPU', int4_compress=False, int8_quant=False):

        if model is None and pretrained_model_path:        
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_path,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_path, 
                trust_remote_code=True
            )
        elif model and tokenizer and pretrained_model_path is None:
            self.model = model
            self.tokenizer = tokenizer

        self.int4_compress = int4_compress
        self.int8_quant = int8_quant
        self.vision_model = VisionModel(model=self.model, ov_model_path=ov_model_path, device=device, int8_quant=self.int8_quant)
        self.vision_mlp_model = VisionMlpModel(model=self.model, ov_model_path=ov_model_path, device=device)

        self.llm_embed_model = LlmEmbdModel(model=self.model, ov_model_path=ov_model_path, device=device)
        self.llm_stateful_model = LlmStatefulModel(model=self.model, tokenizer= self.tokenizer, ov_model_path=ov_model_path, device=device, int4_compress=self.int4_compress)

    def export_vision_to_ov(self):
        self.vision_model.convert_sdpa_ov()
        self.vision_mlp_model.convert_sdpa_ov()
        self.llm_embed_model.convert_sdpa_ov()
        self.llm_stateful_model.convert_sdpa_ov()

class OVInternVLForCausalLM(GenerationMixin):
    def __init__(
        self,
        core=None,
        ov_model_path=None,
        device='CPU',
        int4_compress=False,
        int8_quant=False,
        llm_infer_list=[],
        vision_infer=[],
    ):
        self.ov_model_path = ov_model_path
        self.core = core
        self.ov_device = device
        self.int4_compress = int4_compress
        self.int8_quant = int8_quant

        if int4_compress:
            self.llm_model = core.read_model(Path(f"{ov_model_path}/llm_stateful_int4.xml"))
            self.llm_compiled_model = core.compile_model(self.llm_model, device)
        else:
            self.llm_model = core.read_model(Path(f"{ov_model_path}/llm_stateful.xml"))
            self.llm_compiled_model = core.compile_model(self.llm_model, device)
            
        self.llm_request = self.llm_compiled_model.create_infer_request()

        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.llm_model.inputs)}
        self.output_names = {idx: key for idx, key in enumerate(self.llm_model.outputs)}
        self.key_value_input_names = [key for key in list(self.input_names) if key not in ["beam_idx", "inputs_embeds", "attention_mask", "position_ids"]]
        self.key_value_output_names = [key for key in list(self.output_names)[1:]]
        self.stateful = len(self.key_value_input_names) == 0
        # self.compiled_model = core.compile_model(self.model, device, config = {'INFERENCE_PRECISION_HINT': 'f32'})

        self.config = AutoConfig.from_pretrained(ov_model_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.device = torch.device("cpu")
        self.next_beam_idx = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.past_len = None
        self.main_input_name = "input_ids"
        self._supports_cache_class = False

        self.llm_embd = core.read_model(Path(f"{ov_model_path}/llm_embd.xml"))
        self.llm_embd_compiled_model = core.compile_model(self.llm_embd, device)
        self.llm_embd_request = self.llm_embd_compiled_model.create_infer_request()
        
        self.tokenizer = AutoTokenizer.from_pretrained(ov_model_path, trust_remote_code=True)

        self.vision_model_init()

        self.llm_infer_list = llm_infer_list
        self.vision_infer = vision_infer

        self.conv_template = get_conv_template("phi3-chat")
        self.system_message = self.conv_template.system_message
        image_size = self.config.force_image_size or self.config.vision_config.image_size
        patch_size = self.config.vision_config.patch_size
        self.num_image_token = int((image_size // patch_size) ** 2 * (self.config.downsample_ratio ** 2))
 

    def vision_model_init(self):
        if self.int8_quant:
            self.vision_encoder_model = self.core.read_model(Path(f"{self.ov_model_path}/vision_int8.xml"))
        else:
            self.vision_encoder_model = self.core.read_model(Path(f"{self.ov_model_path}/vision.xml"))
        # self.vision_encoder_compiled_model = self.core.compile_model(self.vision_encoder_model, self.ov_device, config = {'INFERENCE_PRECISION_HINT': 'f32'})
        self.vision_encoder_compiled_model = self.core.compile_model(self.vision_encoder_model, self.ov_device)

        self.vision_encoder_request = self.vision_encoder_compiled_model.create_infer_request()

        self.vision_mlp_model = self.core.read_model(Path(f"{self.ov_model_path}/vision_mlp.xml"))
        self.vision_mlp_compiled_model = self.core.compile_model(self.vision_mlp_model, self.ov_device)
        self.vision_mlp_request = self.vision_mlp_compiled_model.create_infer_request()

        self.vision_pre_process = Preprocess()
        self.vision_middle_process = Postprocess()

    def vision_encoder_run(self, pixel_values=None):
        inputs_dict = {}
        inputs_dict['pixel_values'] = pixel_values
        self.vision_encoder_request.start_async(inputs_dict, share_inputs=True)
        self.vision_encoder_request.wait()
        return torch.from_numpy(self.vision_encoder_request.get_tensor("vision_output").data)
    
    def vision_mlp_run(self, vit_embeds=None):
        inputs_dict = {}
        inputs_dict['vit_embeds'] = vit_embeds
        self.vision_mlp_request.start_async(inputs_dict, share_inputs=True)
        self.vision_mlp_request.wait()
        return torch.from_numpy(self.vision_mlp_request.get_tensor("vit_mlp").data)
    
    def load_image(self, image):
        self.vision_infer.clear()
        vision_pre_start = time.perf_counter()
        pixel_values = self.vision_pre_process.load_image(image)
        vision_pre_end = time.perf_counter()
        vision_pre_time = (vision_pre_end - vision_pre_start) * 1000
        self.vision_infer.append(vision_pre_time)

        return pixel_values

    def vision_model(self, pixel_values):
        encoder_start = time.perf_counter()
        if self.int8_quant:
            ### in MTL platform , we set vision model batch = 1, so we unroll the loop
            vision_output = []
            batch = pixel_values.shape[0]
            for i in range(batch):
                pixel_value = pixel_values[i].unsqueeze(0)
                v_o = self.vision_encoder_run(pixel_values=pixel_value)
                vision_output.append(v_o)
            vision_output = torch.stack(vision_output, 0).squeeze(1)    
        else:
            vision_output = self.vision_encoder_run(pixel_values=pixel_values)
        encoder_end = time.perf_counter()
        vit_embeds = self.vision_middle_process.postprocess(vit_embeds=vision_output)
        mlp_start = time.perf_counter()
        vit_mlp = self.vision_mlp_run(vit_embeds=vit_embeds)
        mlp_end = time.perf_counter()
        encoder_time = (encoder_end - encoder_start) * 1000
        mlp_time = (mlp_end - mlp_start) * 1000
        vision_post_time = (mlp_start - encoder_end) * 1000
        self.vision_infer.append(encoder_time)
        self.vision_infer.append(mlp_time)
        self.vision_infer.append(vision_post_time)
        
        return vit_mlp

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True
    
    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def llm_embd_run(self, input_ids):
        llm_embd_inputs = {}
        llm_embd_inputs['input_ids'] = input_ids

        self.llm_embd_request.start_async(llm_embd_inputs, share_inputs=True)
        self.llm_embd_request.wait()

        return torch.from_numpy(self.llm_embd_request.get_tensor("inputs_embeds").data)

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """General inference method"""
        inputs_dict = {}
        if past_key_values is not None:
            inputs_embeds = self.llm_embd_run(input_ids)
            inputs_dict['inputs_embeds'] = inputs_embeds
        else:
            self.past_len = 0
            self.llm_request.reset_state()
            inputs_dict['inputs_embeds'] = inputs_embeds

        inputs_dict["attention_mask"] = attention_mask
        inputs_dict["position_ids"] = position_ids

        batch_size = inputs_embeds.shape[0]
        if "beam_idx" in self.input_names:
            inputs_dict["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

        # print('attention_mask: ', inputs_dict['attention_mask'].shape)
        # print('position_ids: ', inputs_dict['position_ids'])
        # print('inputs_embeds: ', inputs_dict['inputs_embeds'])
        start = time.perf_counter()
        self.llm_request.start_async(inputs_dict, share_inputs=True)
        self.llm_request.wait()
        end = time.perf_counter()

        generation_time = (end - start) * 1000
        self.llm_infer_list.append(generation_time)

        past_key_values = ((),)
        self.past_len += inputs_dict["inputs_embeds"].shape[1]

        # print('logits: ', self.request.get_tensor("logits").data)
        return CausalLMOutputWithPast(
            loss=None,
            logits=torch.from_numpy(self.llm_request.get_tensor("logits").data),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )   
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            cache_length = past_length = self.past_len
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - self.past_len) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif self.past_len < input_ids.shape[1]:
                input_ids = input_ids[:, self.past_len:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]
        else:
            self.llm_infer_list.clear()

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:    
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    def get_input_embeds(self, pixel_values=None, input_ids=None):
        if pixel_values is not None:
            vit_embeds = self.vision_model(pixel_values)
            input_embeds = self.llm_embd_run(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.llm_embd_run(input_ids)
        
        return input_embeds
    
    def chat(self, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):
        
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template("phi3-chat")
        template.system_message = self.system_message
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        generation_config['eos_token_id'] = eos_token_id

        inputs_emds = self.get_input_embeds(pixel_values, input_ids)

        generation_output = self.generate(
            inputs_embeds=inputs_emds,
            attention_mask=attention_mask,
            **generation_config
        )
        response = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    
