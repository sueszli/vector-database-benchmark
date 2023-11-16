import inspect
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import PIL_INTERPOLATION, is_accelerate_available, is_accelerate_version, logging, replace_example_docstring
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from torchvision.utils import save_image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from .vaehook import VAEHook
logger = logging.get_logger(__name__)
EXAMPLE_DOC_STRING = '\n    Examples:\n        ```py\n        >>> # !pip install opencv-python transformers accelerate\n        >>> from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler\n        >>> from diffusers.utils import load_image\n        >>> import numpy as np\n        >>> import torch\n\n        >>> import cv2\n        >>> from PIL import Image\n\n        >>> # download an image\n        >>> image = load_image(\n        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"\n        ... )\n        >>> image = np.array(image)\n\n        >>> # get canny image\n        >>> image = cv2.Canny(image, 100, 200)\n        >>> image = image[:, :, None]\n        >>> image = np.concatenate([image, image, image], axis=2)\n        >>> canny_image = Image.fromarray(image)\n\n        >>> # load control net and stable diffusion v1-5\n        >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)\n        >>> pipe = StableDiffusionControlNetPipeline.from_pretrained(\n        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16\n        ... )\n\n        >>> # speed up diffusion process with faster scheduler and memory optimization\n        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)\n        >>> # remove following line if xformers is not installed\n        >>> pipe.enable_xformers_memory_efficient_attention()\n\n        >>> pipe.enable_model_cpu_offload()\n\n        >>> # generate image\n        >>> generator = torch.manual_seed(0)\n        >>> image = pipe(\n        ...     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image\n        ... ).images[0]\n        ```\n'

class PixelAwareStableDiffusionPipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    """
    Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. If you set multiple ControlNets
            as a list, the outputs from each ControlNet are added together to create one combined additional
            conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ['safety_checker', 'feature_extractor']

    def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel], scheduler: KarrasDiffusionSchedulers, safety_checker: StableDiffusionSafetyChecker, feature_extractor: CLIPImageProcessor, requires_safety_checker: bool=True):
        if False:
            return 10
        super().__init__()
        if safety_checker is None and requires_safety_checker:
            logger.warning(f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .')
        if safety_checker is not None and feature_extractor is None:
            raise ValueError("Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead.")
        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=safety_checker, feature_extractor=feature_extractor)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256, fast_decoder=False, fast_encoder=False, color_fix=False, vae_to_gpu=True):
        if False:
            i = 10
            return i + 15
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)
        encoder = self.vae.encoder
        decoder = self.vae.decoder
        self.vae.encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def enable_vae_slicing(self):
        if False:
            return 10
        '\n        Enable sliced VAE decoding.\n\n        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several\n        steps. This is useful to save some memory and allow larger batch sizes.\n        '
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        if False:
            i = 10
            return i + 15
        '\n        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to\n        computing decoding in one step.\n        '
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        if False:
            i = 10
            return i + 15
        '\n        Enable tiled VAE decoding.\n\n        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in\n        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.\n        '
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        if False:
            return 10
        '\n        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to\n        computing decoding in one step.\n        '
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if False:
            while True:
                i = 10
        "\n        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,\n        text_encoder, vae, controlnet, and safety checker have their state dicts saved to CPU and then are moved to a\n        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.\n        Note that offloading happens on a submodule basis. Memory savings are higher than with\n        `enable_model_cpu_offload`, but performance is lower.\n        "
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError('Please install accelerate via `pip install accelerate`')
        device = torch.device(f'cuda:{gpu_id}')
        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.controlnet]:
            cpu_offload(cpu_offloaded_model, device)
        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        if False:
            i = 10
            return i + 15
        '\n        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared\n        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`\n        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with\n        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.\n        '
        if is_accelerate_available() and is_accelerate_version('>=', '0.17.0.dev0'):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError('`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.')
        device = torch.device(f'cuda:{gpu_id}')
        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            (_, hook) = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)
        if self.safety_checker is not None:
            (_, hook) = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)
        cpu_offload_with_hook(self.controlnet, device)
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        if False:
            print('Hello World!')
        "\n        Returns the device on which the pipeline's models will be executed. After calling\n        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module\n        hooks.\n        "
        if not hasattr(self.unet, '_hf_hook'):
            return self.device
        for module in self.unet.modules():
            if hasattr(module, '_hf_hook') and hasattr(module._hf_hook, 'execution_device') and (module._hf_hook.execution_device is not None):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None, prompt_embeds: Optional[torch.FloatTensor]=None, negative_prompt_embeds: Optional[torch.FloatTensor]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Encodes the prompt into text encoder hidden states.\n\n        Args:\n             prompt (`str` or `List[str]`, *optional*):\n                prompt to be encoded\n            device: (`torch.device`):\n                torch device\n            num_images_per_prompt (`int`):\n                number of images that should be generated per prompt\n            do_classifier_free_guidance (`bool`):\n                whether to use classifier free guidance or not\n            negative_prompt (`str` or `List[str]`, *optional*):\n                The prompt or prompts not to guide the image generation. If not defined, one has to pass\n                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is\n                less than `1`).\n            prompt_embeds (`torch.FloatTensor`, *optional*):\n                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not\n                provided, text embeddings will be generated from `prompt` input argument.\n            negative_prompt_embeds (`torch.FloatTensor`, *optional*):\n                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt\n                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input\n                argument.\n        '
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
            text_inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding='longest', return_tensors='pt').input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and (not torch.equal(text_input_ids, untruncated_ids)):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1:-1])
                logger.warning(f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}')
            if hasattr(self.text_encoder.config, 'use_attention_mask') and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        (bs_embed, seq_len, _) = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [''] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.')
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.')
            else:
                uncond_tokens = negative_prompt
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(uncond_tokens, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
            if hasattr(self.text_encoder.config, 'use_attention_mask') and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device), attention_mask=attention_mask)
            negative_prompt_embeds = negative_prompt_embeds[0]
        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    def run_safety_checker(self, image, device, dtype):
        if False:
            print('Hello World!')
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type='pil')
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors='pt').to(device)
            (image, has_nsfw_concept) = self.safety_checker(images=image, clip_input=safety_checker_input.pixel_values.to(dtype))
        return (image, has_nsfw_concept)

    def decode_latents(self, latents):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('The decode_latents method is deprecated and will be removed in a future version. Please use VaeImageProcessor instead', FutureWarning)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        if False:
            i = 10
            return i + 15
        accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta
        accepts_generator = 'generator' in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, image, height, width, callback_steps, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None, controlnet_conditioning_scale=1.0):
        if False:
            i = 10
            return i + 15
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f'`height` and `width` have to be divisible by 8 but are {height} and {width}.')
        flag = callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        if callback_steps is None or flag:
            raise ValueError(f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.')
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.')
        elif prompt is None and prompt_embeds is None:
            raise ValueError('Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.')
        elif prompt is not None and (not isinstance(prompt, str) and (not isinstance(prompt, list))):
            raise ValueError(f'`prompt` has to be of type `str` or `list` but is {type(prompt)}')
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(f'Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two.')
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(f'`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}.')
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(f'You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)} prompts. The conditionings will be fixed across the prompts.')
        is_compiled = hasattr(F, 'scaled_dot_product_attention') and isinstance(self.controlnet, torch._dynamo.eval_frame.OptimizedModule)
        if isinstance(self.controlnet, ControlNetModel) or (is_compiled and isinstance(self.controlnet._orig_mod, ControlNetModel)):
            self.check_image(image, prompt, prompt_embeds)
        elif isinstance(self.controlnet, MultiControlNetModel) or (is_compiled and isinstance(self.controlnet._orig_mod, MultiControlNetModel)):
            if not isinstance(image, list):
                raise TypeError('For multiple controlnets: `image` must be type `list`')
            elif any((isinstance(i, list) for i in image)):
                raise ValueError('A single batch of multiple conditionings are supported at the moment.')
            elif len(image) != len(self.controlnet.nets):
                raise ValueError('For multiple controlnets: `image` must have the same length as the number of controlnets.')
            for image_ in image:
                self.check_image(image_, prompt, prompt_embeds)
        else:
            assert False
        if isinstance(self.controlnet, ControlNetModel) or (is_compiled and isinstance(self.controlnet._orig_mod, ControlNetModel)):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError('For single controlnet: `controlnet_conditioning_scale` must be type `float`.')
        elif isinstance(self.controlnet, MultiControlNetModel) or (is_compiled and isinstance(self.controlnet._orig_mod, MultiControlNetModel)):
            if isinstance(controlnet_conditioning_scale, list):
                if any((isinstance(i, list) for i in controlnet_conditioning_scale)):
                    raise ValueError('A single batch of multiple conditionings are supported at the moment.')
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(self.controlnet.nets):
                raise ValueError('For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`,it must have the same length as the number of controlnets')
        else:
            assert False

    def check_image(self, image, prompt, prompt_embeds):
        if False:
            for i in range(10):
                print('nop')
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        if not image_is_pil and (not image_is_tensor) and (not image_is_pil_list) and (not image_is_tensor_list):
            raise TypeError('image must be passed and be one of PIL image, torch tensor, list of PIL images,or list of torch tensors')
        if image_is_pil:
            image_batch_size = 1
        elif image_is_tensor:
            image_batch_size = image.shape[0]
        elif image_is_pil_list:
            image_batch_size = len(image)
        elif image_is_tensor_list:
            image_batch_size = len(image)
        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]
        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(f'If image batch size is not 1, image batch size must be same as prompt batch size.                     image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}')

    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype, do_classifier_free_guidance=False, guess_mode=False):
        if False:
            return 10
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]
            if isinstance(image[0], PIL.Image.Image):
                images = []
                for image_ in image:
                    image_ = image_.convert('RGB')
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)
                image = images
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)
        image_batch_size = image.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt
        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)
        if do_classifier_free_guidance and (not guess_mode):
            image = torch.cat([image] * 2)
        return image

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        if False:
            for i in range(10):
                print('nop')
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.')
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _default_height_width(self, height, width, image):
        if False:
            print('Hello World!')
        while isinstance(image, list):
            image = image[0]
        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]
            height = height // 8 * 8
        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]
            width = width // 8 * 8
        return (height, width)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], safe_serialization: bool=False, variant: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        if isinstance(self.controlnet, ControlNetModel):
            super().save_pretrained(save_directory, safe_serialization, variant)
        else:
            raise NotImplementedError('Currently, the `save_pretrained()` is not implemented for Multi-ControlNet.')

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        if False:
            return 10
        'Generates a gaussian mask of weights for tile contributions'
        from numpy import pi, exp, sqrt
        import numpy as np
        latent_width = tile_width
        latent_height = tile_height
        var = 0.01
        midpoint = (latent_width - 1) / 2
        x_probs = []
        for x in range(latent_width):
            tmp = -(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)
            tmp = exp(tmp) / sqrt(2 * pi * var)
            x_probs.append(tmp)
        midpoint = latent_height / 2
        y_probs = []
        for y in range(latent_height):
            tmp = -(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)
            tmp = exp(tmp) / sqrt(2 * pi * var)
            y_probs.append(tmp)
        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(self, prompt: Union[str, List[str]]=None, image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]]=None, height: Optional[int]=None, width: Optional[int]=None, num_inference_steps: int=50, guidance_scale: float=7.5, negative_prompt: Optional[Union[str, List[str]]]=None, num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents: Optional[torch.FloatTensor]=None, prompt_embeds: Optional[torch.FloatTensor]=None, negative_prompt_embeds: Optional[torch.FloatTensor]=None, output_type: Optional[str]='pil', return_dict: bool=True, callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None, callback_steps: int=1, cross_attention_kwargs: Optional[Dict[str, Any]]=None, fg_mask: Optional[torch.FloatTensor]=None, conditioning_scale_fg: Union[float, List[float]]=1.0, conditioning_scale_bg: Union[float, List[float]]=1.0, guess_mode: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Function invoked when calling the pipeline for generation.\n\n        Args:\n            prompt (`str` or `List[str]`, *optional*):\n                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.\n                instead.\n            image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`,\n                    `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):\n                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If\n                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can\n                also be accepted as an image. The dimensions of the output image defaults to `image`\'s dimensions. If\n                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are\n                specified in init, images must be passed as a list such that each element of the list can be correctly\n                batched for input to a single controlnet.\n            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):\n                The height in pixels of the generated image.\n            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):\n                The width in pixels of the generated image.\n            num_inference_steps (`int`, *optional*, defaults to 50):\n                The number of denoising steps. More denoising steps usually lead to a higher quality image at the\n                expense of slower inference.\n            guidance_scale (`float`, *optional*, defaults to 7.5):\n                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).\n                `guidance_scale` is defined as `w` of equation 2. of [Imagen\n                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >\n                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,\n                usually at the expense of lower image quality.\n            negative_prompt (`str` or `List[str]`, *optional*):\n                The prompt or prompts not to guide the image generation. If not defined, one has to pass\n                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is\n                less than `1`).\n            num_images_per_prompt (`int`, *optional*, defaults to 1):\n                The number of images to generate per prompt.\n            eta (`float`, *optional*, defaults to 0.0):\n                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to\n                [`schedulers.DDIMScheduler`], will be ignored for others.\n            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):\n                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)\n                to make generation deterministic.\n            latents (`torch.FloatTensor`, *optional*):\n                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image\n                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents\n                tensor will ge generated by sampling using the supplied random `generator`.\n            prompt_embeds (`torch.FloatTensor`, *optional*):\n                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not\n                provided, text embeddings will be generated from `prompt` input argument.\n            negative_prompt_embeds (`torch.FloatTensor`, *optional*):\n                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt\n                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input\n                argument.\n            output_type (`str`, *optional*, defaults to `"pil"`):\n                The output format of the generate image. Choose between\n                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.\n            return_dict (`bool`, *optional*, defaults to `True`):\n                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a\n                plain tuple.\n            callback (`Callable`, *optional*):\n                A function that will be called every `callback_steps` steps during inference. The function will be\n                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.\n            callback_steps (`int`, *optional*, defaults to 1):\n                The frequency at which the `callback` function will be called. If not specified, the callback will be\n                called at every step.\n            cross_attention_kwargs (`dict`, *optional*):\n                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under\n                `self.processor` in\n                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).\n            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):\n                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added\n                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the\n                corresponding scale as a list.\n            guess_mode (`bool`, *optional*, defaults to `False`):\n                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if\n                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.\n\n        Examples:\n\n        Returns:\n            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:\n            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.\n            When returning a tuple, the first element is a list with the generated images, and the second element is a\n            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"\n            (nsfw) content, according to the `safety_checker`.\n        '
        (height, width) = self._default_height_width(height, width, image)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        guess_mode = guess_mode
        prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds)
        image = self.prepare_image(image=image, width=width, height=height, batch_size=batch_size * num_images_per_prompt, num_images_per_prompt=num_images_per_prompt, device=device, dtype=controlnet.dtype, do_classifier_free_guidance=do_classifier_free_guidance, guess_mode=guess_mode)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, latents)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for (i, t) in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if guess_mode and do_classifier_free_guidance:
                    controlnet_latent_model_input = latents
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    controlnet_latent_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                (_, _, h, w) = latent_model_input.size()
                (tile_size, tile_overlap) = (120, 32)
                if h < tile_size and w < tile_size:
                    if image is not None:
                        (rgbs, down_block_res_samples, mid_block_res_sample) = self.controlnet(controlnet_latent_model_input, t, encoder_hidden_states=controlnet_prompt_embeds, controlnet_cond=image, fg_mask=fg_mask, conditioning_scale_fg=conditioning_scale_fg, conditioning_scale_bg=conditioning_scale_bg, guess_mode=guess_mode, return_dict=False)
                    else:
                        (down_block_res_samples, mid_block_res_sample) = ([None] * 10, [None] * 10)
                    if guess_mode and do_classifier_free_guidance:
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attention_kwargs, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample, return_dict=False)[0]
                else:
                    tile_size = min(tile_size, min(h, w))
                    tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(latent_model_input.device)
                    grid_rows = 0
                    cur_x = 0
                    while cur_x < latent_model_input.size(-1):
                        cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                        grid_rows += 1
                    grid_cols = 0
                    cur_y = 0
                    while cur_y < latent_model_input.size(-2):
                        cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                        grid_cols += 1
                    input_list = []
                    cond_list = []
                    img_list = []
                    fg_mask_list = []
                    noise_preds = []
                    for row in range(grid_rows):
                        for col in range(grid_cols):
                            if col < grid_cols - 1 or row < grid_rows - 1:
                                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                            if row == grid_rows - 1:
                                ofs_x = w - tile_size
                            if col == grid_cols - 1:
                                ofs_y = h - tile_size
                            input_start_x = ofs_x
                            input_end_x = ofs_x + tile_size
                            input_start_y = ofs_y
                            input_end_y = ofs_y + tile_size
                            input_tile = latent_model_input[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                            input_list.append(input_tile)
                            cond_tile = controlnet_latent_model_input[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                            cond_list.append(cond_tile)
                            img_tile = image[:, :, input_start_y * 8:input_end_y * 8, input_start_x * 8:input_end_x * 8]
                            img_list.append(img_tile)
                            if fg_mask is not None:
                                fg_mask_tile = fg_mask[:, :, input_start_y * 8:input_end_y * 8, input_start_x * 8:input_end_x * 8]
                                fg_mask_list.append(fg_mask_tile)
                            if len(input_list) == batch_size or col == grid_cols - 1:
                                input_list_t = torch.cat(input_list, dim=0)
                                cond_list_t = torch.cat(cond_list, dim=0)
                                img_list_t = torch.cat(img_list, dim=0)
                                if fg_mask is not None:
                                    fg_mask_list_t = torch.cat(fg_mask_list, dim=0)
                                else:
                                    fg_mask_list_t = None
                                (_, down_block_res_samples, mid_block_res_sample) = self.controlnet(cond_list_t, t, encoder_hidden_states=controlnet_prompt_embeds, controlnet_cond=img_list_t, fg_mask=fg_mask_list_t, conditioning_scale_fg=conditioning_scale_fg, conditioning_scale_bg=conditioning_scale_bg, guess_mode=guess_mode, return_dict=False)
                                if guess_mode and do_classifier_free_guidance:
                                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                                model_out = self.unet(input_list_t, t, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attention_kwargs, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample, return_dict=False)[0]
                                input_list = []
                                cond_list = []
                                img_list = []
                                fg_mask_list = []
                            noise_preds.append(model_out)
                    noise_pred = torch.zeros(latent_model_input.shape, device=latent_model_input.device)
                    contributors = torch.zeros(latent_model_input.shape, device=latent_model_input.device)
                    for row in range(grid_rows):
                        for col in range(grid_cols):
                            if col < grid_cols - 1 or row < grid_rows - 1:
                                ofs_x = max(row * tile_size - tile_overlap * row, 0)
                                ofs_y = max(col * tile_size - tile_overlap * col, 0)
                            if row == grid_rows - 1:
                                ofs_x = w - tile_size
                            if col == grid_cols - 1:
                                ofs_y = h - tile_size
                            input_start_x = ofs_x
                            input_end_x = ofs_x + tile_size
                            input_start_y = ofs_y
                            input_end_y = ofs_y + tile_size
                            noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row * grid_cols + col] * tile_weights
                            contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
                    noise_pred /= contributors
                if do_classifier_free_guidance:
                    (noise_pred_uncond, noise_pred_text) = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                flag = i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                if i == len(timesteps) - 1 or flag:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        if hasattr(self, 'final_offload_hook') and self.final_offload_hook is not None:
            self.unet.to('cpu')
            self.controlnet.to('cpu')
            torch.cuda.empty_cache()
        has_nsfw_concept = None
        if not output_type == 'latent':
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            has_nsfw_concept = None
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        if hasattr(self, 'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)