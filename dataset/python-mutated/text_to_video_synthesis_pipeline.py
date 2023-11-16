import os
import tempfile
from typing import Any, Dict, Optional
import cv2
import torch
from einops import rearrange
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()

@PIPELINES.register_module(Tasks.text_to_video_synthesis, module_name=Pipelines.text_to_video_synthesis)
class TextToVideoSynthesisPipeline(Pipeline):
    """ Text To Video Synthesis Pipeline.

    Examples:
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.outputs import OutputKeys

    >>> p = pipeline('text-to-video-synthesis', 'damo/text-to-video-synthesis')
    >>> test_text = {
    >>>         'text': 'A panda eating bamboo on a rock.',
    >>>     }
    >>> p(test_text,)

    >>>  {OutputKeys.OUTPUT_VIDEO: path-to-the-generated-video}
    >>>
    """

    def __init__(self, model: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        use `model` to create a kws pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, **kwargs)

    def preprocess(self, input: Input, **preprocess_params) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        self.model.clip_encoder.to(self.model.device)
        text_emb = self.model.clip_encoder(input['text'])
        text_emb_zero = self.model.clip_encoder('')
        if self.model.config.model.model_args.tiny_gpu == 1:
            self.model.clip_encoder.to('cpu')
        out_height = input['height'] if 'height' in input else 256
        out_width = input['width'] if 'height' in input else 256
        return {'text_emb': text_emb, 'text_emb_zero': text_emb_zero, 'out_height': out_height, 'out_width': out_width}

    def forward(self, input: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        video = self.model(input)
        return {'video': video}

    def postprocess(self, inputs: Dict[str, Any], **post_params) -> Dict[str, Any]:
        if False:
            return 10
        video = tensor2vid(inputs['video'])
        output_video_path = post_params.get('output_video', None)
        temp_video_file = False
        if output_video_path is None:
            output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
            temp_video_file = True
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        (h, w, c) = video[0].shape
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=8, frameSize=(w, h))
        for i in range(len(video)):
            img = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
            video_writer.write(img)
        video_writer.release()
        if temp_video_file:
            video_file_content = b''
            with open(output_video_path, 'rb') as f:
                video_file_content = f.read()
            os.remove(output_video_path)
            return {OutputKeys.OUTPUT_VIDEO: video_file_content}
        else:
            return {OutputKeys.OUTPUT_VIDEO: output_video_path}

def tensor2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    if False:
        while True:
            i = 10
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    images = rearrange(video, 'i c f h w -> f h (i w) c')
    images = images.unbind(dim=0)
    images = [(image.numpy() * 255).astype('uint8') for image in images]
    return images