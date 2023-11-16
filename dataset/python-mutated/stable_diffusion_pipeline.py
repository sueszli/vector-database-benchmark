import os
from typing import Any, Dict, Optional
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline
from PIL import Image
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.multi_modal.diffusers_wrapped.diffusers_pipeline import DiffusersPipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.import_utils import is_swift_available

@PIPELINES.register_module(Tasks.text_to_image_synthesis, module_name=Pipelines.diffusers_stable_diffusion)
class StableDiffusionPipeline(DiffusersPipeline):

    def __init__(self, model: str, lora_dir: str=None, custom_dir: str=None, modifier_token: str=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        use `model` to create a stable diffusion pipeline\n        Args:\n            model: model id on modelscope hub or local model dir.\n            lora_dir: lora weight dir for unet.\n            custom_dir: custom diffusion weight dir for unet.\n            modifier_token: token to use as a modifier for the concept of custom diffusion.\n            use_safetensors: load safetensors weights.\n            use_swift: Whether to use swift lora dir for unet.\n        '
        use_safetensors = kwargs.pop('use_safetensors', False)
        torch_type = kwargs.pop('torch_type', torch.float32)
        use_swift = kwargs.pop('use_swift', False)
        if custom_dir is None and modifier_token is not None:
            raise ValueError('custom_dir is None but modifier_token is not None')
        elif custom_dir is not None and modifier_token is None:
            raise ValueError('modifier_token is None but custom_dir is not None')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline = DiffusionPipeline.from_pretrained(model, use_safetensors=use_safetensors, torch_dtype=torch_type)
        self.pipeline = self.pipeline.to(self.device)
        if lora_dir is not None:
            assert os.path.exists(lora_dir), f"{lora_dir} isn't exist"
            if use_swift:
                if not is_swift_available():
                    raise ValueError('Please install swift by `pip install ms-swift` to use efficient_tuners.')
                from swift import Swift
                self.pipeline.unet = Swift.from_pretrained(self.pipeline.unet, lora_dir)
            else:
                self.pipeline.unet.load_attn_procs(lora_dir)
        if custom_dir is not None:
            assert os.path.exists(custom_dir), f"{custom_dir} isn't exist"
            self.pipeline.unet.load_attn_procs(custom_dir, weight_name='pytorch_custom_diffusion_weights.bin')
            modifier_token = modifier_token.split('+')
            for modifier_token_name in modifier_token:
                self.pipeline.load_textual_inversion(custom_dir, weight_name=f'{modifier_token_name}.bin')

    def preprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if False:
            return 10
        return inputs

    def forward(self, inputs: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            return 10
        '\n        Inputs Args:\n            prompt (`str` or `List[str]`, *optional*):\n                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.\n                instead.\n            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):\n                The height in pixels of the generated image.\n            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):\n                The width in pixels of the generated image.\n            num_inference_steps (`int`, *optional*, defaults to 50):\n                The number of denoising steps. More denoising steps usually lead to a higher quality image at the\n                expense of slower inference.\n            guidance_scale (`float`, *optional*, defaults to 7.5):\n                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).\n                `guidance_scale` is defined as `w` of equation 2. of [Imagen\n                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >\n                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,\n                usually at the expense of lower image quality.\n            negative_prompt (`str` or `List[str]`, *optional*):\n                The prompt or prompts not to guide the image generation. If not defined, one has to pass\n                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is\n                less than `1`).\n            num_images_per_prompt (`int`, *optional*, defaults to 1):\n                The number of images to generate per prompt.\n            eta (`float`, *optional*, defaults to 0.0):\n                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to\n                [`schedulers.DDIMScheduler`], will be ignored for others.\n            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):\n                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)\n                to make generation deterministic.\n            latents (`torch.FloatTensor`, *optional*):\n                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image\n                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents\n                tensor will ge generated by sampling using the supplied random `generator`.\n            prompt_embeds (`torch.FloatTensor`, *optional*):\n                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not\n                provided, text embeddings will be generated from `prompt` input argument.\n            negative_prompt_embeds (`torch.FloatTensor`, *optional*):\n                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt\n                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input\n                argument.\n            output_type (`str`, *optional*, defaults to `"pil"`):\n                The output format of the generate image. Choose between\n                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.\n            return_dict (`bool`, *optional*, defaults to `True`):\n                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a\n                plain tuple.\n            callback (`Callable`, *optional*):\n                A function that will be called every `callback_steps` steps during inference. The function will be\n                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.\n            callback_steps (`int`, *optional*, defaults to 1):\n                The frequency at which the `callback` function will be called. If not specified, the callback will be\n                called at every step.\n        '
        if not isinstance(inputs, dict):
            raise ValueError(f'Expected the input to be a dictionary, but got {type(input)}')
        if 'text' not in inputs:
            raise ValueError('input should contain "text", but not found')
        images = self.pipeline(prompt=inputs.get('text'), height=inputs.get('height'), width=inputs.get('width'), num_inference_steps=inputs.get('num_inference_steps', 50), guidance_scale=inputs.get('guidance_scale', 7.5), negative_prompt=inputs.get('negative_prompt'), num_images_per_prompt=inputs.get('num_images_per_prompt', 1), eta=inputs.get('eta', 0.0), generator=inputs.get('generator'), latents=inputs.get('latents'), output_type=inputs.get('output_type', 'pil'), return_dict=inputs.get('return_dict', True), callback=inputs.get('callback'), callback_steps=inputs.get('callback_steps', 1))
        return images

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        images = []
        for img in inputs.images:
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img)
        return {OutputKeys.OUTPUT_IMGS: images}