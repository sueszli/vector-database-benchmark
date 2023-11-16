import os
from typing import Any, Callable, Dict, List, Optional, Union
import cv2
import numpy as np
import PIL
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL, ControlNetModel, DiffusionPipeline, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import deprecate, is_accelerate_available, is_accelerate_version, is_compiled_module, logging, randn_tensor, replace_example_docstring
from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj
from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.text_texture_generation.lib2.camera import *
from modelscope.models.cv.text_texture_generation.lib2.init_view import *
from modelscope.models.cv.text_texture_generation.utils import *
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()
EXAMPLE_DOC_STRING = '\n    Examples:\n        ```py\n        >>> from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler\n        >>> from diffusers.utils import load_image\n        >>> import numpy as np\n        >>> import torch\n\n        >>> init_image = load_image(image_path)\n        >>> init_image = init_image.resize((512, 512))\n        >>> generator = torch.Generator(device="cpu").manual_seed(1)\n        >>> mask_image = load_image(mask_path)\n        >>> mask_image = mask_image.resize((512, 512))\n        >>> def make_inpaint_condition(image, image_mask):\n        ...     image = np.array(image.convert("RGB")).astype(np.float32) / 255.0\n        ...     image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0\n        ...     assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"\n        ...     image[image_mask > 0.5] = -1.0  # set as masked pixel\n        ...     image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)\n        ...     image = torch.from_numpy(image)\n        ...     return image\n        >>> control_image = make_inpaint_condition(init_image, mask_image)\n        >>> controlnet = ControlNetModel.from_pretrained(\n        ...     "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16\n        ... )\n        >>> pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(\n        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16\n        ... )\n        >>> pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)\n        >>> pipe.enable_model_cpu_offload()\n        >>> image = pipe(\n        ...     "a handsome man with ray-ban sunglasses",\n        ...     num_inference_steps=20,\n        ...     generator=generator,\n        ...     eta=1.0,\n        ...     image=init_image,\n        ...     mask_image=mask_image,\n        ...     control_image=control_image,\n        ... ).images[0]\n        ```\n'

@MODELS.register_module(Tasks.text_texture_generation, module_name=Models.text_texture_generation)
class Tex2Texture(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        if False:
            print('Hello World!')
        'The Tex2Texture is modified based on TEXTure and Text2Tex, publicly available at\n                https://github.com/TEXTurePaper/TEXTurePaper &\n                https://github.com/daveredrum/Text2Tex\n        Args:\n            model_dir: the root directory of the model files\n        '
        super().__init__(*args, model_dir=model_dir, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info('Use GPU: {}'.format(self.device))
        else:
            print('no gpu avaiable')
            exit()
        model_path = model_dir + '/base_model/'
        controlmodel_path = model_dir + '/control_model/'
        inpaintmodel_path = model_dir + '/inpaint_model/'
        torch_dtype = kwargs.get('torch_dtype', torch.float16)
        self.controlnet = ControlNetModel.from_pretrained(controlmodel_path, torch_dtype=torch_dtype).to(self.device)
        self.inpaintmodel = StableDiffusionInpaintPipeline.from_pretrained(inpaintmodel_path, torch_dtype=torch_dtype).to(self.device)
        self.pipe = StableDiffusionControlinpaintPipeline.from_pretrained(model_path, controlnet=self.controlnet, torch_dtype=torch_dtype).to(self.device)
        logger.info('model load over')

    def init_mesh(self, mesh_path):
        if False:
            i = 10
            return i + 15
        (verts, faces, aux) = load_obj(mesh_path, device=self.device)
        mesh = load_objs_as_meshes([mesh_path], device=self.device)
        return (mesh, verts, faces, aux)

    def normalize_mesh(self, mesh):
        if False:
            i = 10
            return i + 15
        bbox = mesh.get_bounding_boxes()
        num_verts = mesh.verts_packed().shape[0]
        mesh_center = bbox.mean(dim=2).repeat(num_verts, 1)
        mesh = mesh.offset_verts(-mesh_center)
        lens = bbox[0, :, 1] - bbox[0, :, 0]
        max_len = lens.max()
        scale = 0.9 / max_len
        scale = scale.unsqueeze(0).repeat(num_verts)
        new_mesh = mesh.scale_verts(scale)
        return (new_mesh.verts_packed(), new_mesh, mesh_center, scale)

    def save_normalized_obj(self, verts, faces, aux, path='normalized.obj'):
        if False:
            return 10
        print('=> saving normalized mesh file...')
        obj_path = path
        save_obj(obj_path, verts=verts, faces=faces.verts_idx, decimal_places=5, verts_uvs=aux.verts_uvs, faces_uvs=faces.textures_idx, texture_map=aux.texture_images[list(aux.texture_images.keys())[0]])

    def mesh_normalized(self, mesh_path, save_path='normalized.obj'):
        if False:
            return 10
        (mesh, verts, faces, aux) = self.init_mesh(mesh_path)
        (verts, mesh, mesh_center, scale) = self.normalize_mesh(mesh)
        self.save_normalized_obj(verts, faces, aux, save_path)
        return (mesh, verts, faces, aux, mesh_center, scale)

def prepare_mask_and_masked_image(image, mask, height, width, return_image=False):
    if False:
        i = 10
        return i + 15
    if image is None:
        raise ValueError('`image` input cannot be undefined.')
    if mask is None:
        raise ValueError('`mask_image` input cannot be undefined.')
    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f'`image` is a torch.Tensor but `mask` (type: {type(mask)} is not')
        if image.ndim == 3:
            assert image.shape[0] == 3, 'Image outside a batch should be of shape (3, H, W)'
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)
            else:
                mask = mask.unsqueeze(1)
        assert image.ndim == 4 and mask.ndim == 4, 'Image and Mask must have 4 dimensions'
        assert image.shape[-2:] == mask.shape[-2:], 'Image and Mask must have the same spatial dimensions'
        assert image.shape[0] == mask.shape[0], 'Image and Mask must have the same batch size'
        if image.min() < -1 or image.max() > 1:
            raise ValueError('Image should be in [-1, 1] range')
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError('Mask should be in [0, 1] range')
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f'`mask` is a torch.Tensor but `image` (type: {type(image)} is not')
    else:
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB'))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]
        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert('L'))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
    masked_image = image * (mask < 0.5)
    if return_image:
        return (mask, masked_image, image)
    return (mask, masked_image)

class StableDiffusionControlinpaintPipeline(StableDiffusionControlNetInpaintPipeline):

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(self, prompt: Union[str, List[str]]=None, image: Union[torch.Tensor, PIL.Image.Image]=None, mask_image: Union[torch.Tensor, PIL.Image.Image]=None, control_image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]=None, height: Optional[int]=None, width: Optional[int]=None, strength: float=1.0, num_inference_steps: int=50, guidance_scale: float=7.5, negative_prompt: Optional[Union[str, List[str]]]=None, num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents: Optional[torch.FloatTensor]=None, prompt_embeds: Optional[torch.FloatTensor]=None, negative_prompt_embeds: Optional[torch.FloatTensor]=None, output_type: Optional[str]='pil', return_dict: bool=True, callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None, callback_steps: int=1, cross_attention_kwargs: Optional[Dict[str, Any]]=None, controlnet_conditioning_scale: Union[float, List[float]]=0.5, guess_mode: bool=False):
        if False:
            print('Hello World!')
        '\n        Function invoked when calling the pipeline for generation.\n\n        Args:\n            prompt (`str` or `List[str]`, *optional*):\n                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.\n                instead.\n            image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`,\n                    `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):\n                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If\n                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can\n                also be accepted as an image. The dimensions of the output image defaults to `image`\'s dimensions. If\n                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are\n                specified in init, images must be passed as a list such that each element of the list can be correctly\n                batched for input to a single controlnet.\n            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):\n                The height in pixels of the generated image.\n            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):\n                The width in pixels of the generated image.\n            strength (`float`, *optional*, defaults to 1.):\n                Conceptually, indicates how much to transform the masked portion of the reference `image`. Must be\n                between 0 and 1. `image` will be used as a starting point, adding more noise to it the larger the\n                `strength`. The number of denoising steps depends on the amount of noise initially added. When\n                `strength` is 1, added noise will be maximum and the denoising process will run for the full number of\n                iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores the masked\n                portion of the reference `image`.\n            num_inference_steps (`int`, *optional*, defaults to 50):\n                The number of denoising steps. More denoising steps usually lead to a higher quality image at the\n                expense of slower inference.\n            guidance_scale (`float`, *optional*, defaults to 7.5):\n                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).\n                `guidance_scale` is defined as `w` of equation 2. of [Imagen\n                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >\n                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,\n                usually at the expense of lower image quality.\n            negative_prompt (`str` or `List[str]`, *optional*):\n                The prompt or prompts not to guide the image generation. If not defined, one has to pass\n                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is\n                less than `1`).\n            num_images_per_prompt (`int`, *optional*, defaults to 1):\n                The number of images to generate per prompt.\n            eta (`float`, *optional*, defaults to 0.0):\n                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to\n                [`schedulers.DDIMScheduler`], will be ignored for others.\n            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):\n                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)\n                to make generation deterministic.\n            latents (`torch.FloatTensor`, *optional*):\n                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image\n                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents\n                tensor will ge generated by sampling using the supplied random `generator`.\n            prompt_embeds (`torch.FloatTensor`, *optional*):\n                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not\n                provided, text embeddings will be generated from `prompt` input argument.\n            negative_prompt_embeds (`torch.FloatTensor`, *optional*):\n                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt\n                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input\n                argument.\n            output_type (`str`, *optional*, defaults to `"pil"`):\n                The output format of the generate image. Choose between\n                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.\n            return_dict (`bool`, *optional*, defaults to `True`):\n                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a\n                plain tuple.\n            callback (`Callable`, *optional*):\n                A function that will be called every `callback_steps` steps during inference. The function will be\n                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.\n            callback_steps (`int`, *optional*, defaults to 1):\n                The frequency at which the `callback` function will be called. If not specified, the callback will be\n                called at every step.\n            cross_attention_kwargs (`dict`, *optional*):\n                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under\n                `self.processor` in\n                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).\n            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 0.5):\n                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added\n                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the\n                corresponding scale as a list. Note that by default, we use a smaller conditioning scale for inpainting\n                than for [`~StableDiffusionControlNetPipeline.__call__`].\n            guess_mode (`bool`, *optional*, defaults to `False`):\n                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if\n                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.\n\n        Examples:\n\n        Returns:\n            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:\n            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.\n            When returning a tuple, the first element is a list with the generated images, and the second element is a\n            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"\n            (nsfw) content, according to the `safety_checker`.\n        '
        (height, width) = self._default_height_width(height, width, image)
        self.check_inputs(prompt, control_image, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, controlnet_conditioning_scale)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)
        global_pool_conditions = controlnet.config.global_pool_conditions if isinstance(controlnet, ControlNetModel) else controlnet.nets[0].config.global_pool_conditions
        guess_mode = guess_mode or global_pool_conditions
        text_encoder_lora_scale = cross_attention_kwargs.get('scale', None) if cross_attention_kwargs is not None else None
        prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, lora_scale=text_encoder_lora_scale)
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(image=control_image, width=width, height=height, batch_size=batch_size * num_images_per_prompt, num_images_per_prompt=num_images_per_prompt, device=device, dtype=controlnet.dtype, do_classifier_free_guidance=do_classifier_free_guidance, guess_mode=guess_mode)
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []
            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(image=control_image_, width=width, height=height, batch_size=batch_size * num_images_per_prompt, num_images_per_prompt=num_images_per_prompt, device=device, dtype=controlnet.dtype, do_classifier_free_guidance=do_classifier_free_guidance, guess_mode=guess_mode)
                control_images.append(control_image_)
            control_image = control_images
        else:
            assert False
        (mask, masked_image, init_image) = prepare_mask_and_masked_image(image, mask_image, height, width, return_image=True)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        (timesteps, num_inference_steps) = self.get_timesteps(num_inference_steps=num_inference_steps, strength=strength, device=device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        is_strength_max = strength == 1.0
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4
        latents_outputs = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, latents, image=init_image, timestep=latent_timestep, is_strength_max=is_strength_max, return_noise=True, return_image_latents=return_image_latents)
        if return_image_latents:
            (latents, noise, image_latents) = latents_outputs
        else:
            (latents, noise) = latents_outputs
        (mask, masked_image_latents) = self.prepare_mask_latents(mask, masked_image, batch_size * num_images_per_prompt, height, width, prompt_embeds.dtype, device, generator, do_classifier_free_guidance)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for (i, t) in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if guess_mode and do_classifier_free_guidance:
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                (down_block_res_samples, mid_block_res_sample) = self.controlnet(control_model_input, t, encoder_hidden_states=controlnet_prompt_embeds, controlnet_cond=control_image, conditioning_scale=controlnet_conditioning_scale, guess_mode=guess_mode, return_dict=False)
                if guess_mode and do_classifier_free_guidance:
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attention_kwargs, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample, return_dict=False)[0]
                if do_classifier_free_guidance:
                    (noise_pred_uncond, noise_pred_text) = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if num_channels_unet == 4:
                    init_latents_proper = image_latents[:1]
                    init_mask = mask[:1]
                    if i < len(timesteps) - 1:
                        init_latents_proper = self.scheduler.add_noise(init_latents_proper, noise, torch.tensor([t]))
                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        if hasattr(self, 'final_offload_hook') and self.final_offload_hook is not None:
            self.unet.to('cpu')
            self.controlnet.to('cpu')
            torch.cuda.empty_cache()
        if not output_type == 'latent':
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            (image, has_nsfw_concept) = self.run_safety_checker(image, device, prompt_embeds.dtype)
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