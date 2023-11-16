from io import BytesIO
from typing import List, Union
import requests
from ..utils import add_end_docstrings, is_decord_available, is_torch_available, logging, requires_backends
from .base import PIPELINE_INIT_ARGS, Pipeline
if is_decord_available():
    import numpy as np
    from decord import VideoReader
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES
logger = logging.get_logger(__name__)

@add_end_docstrings(PIPELINE_INIT_ARGS)
class VideoClassificationPipeline(Pipeline):
    """
    Video classification pipeline using any `AutoModelForVideoClassification`. This pipeline predicts the class of a
    video.

    This video classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"video-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=video-classification).
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        requires_backends(self, 'decord')
        self.check_model_type(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES)

    def _sanitize_parameters(self, top_k=None, num_frames=None, frame_sampling_rate=None):
        if False:
            print('Hello World!')
        preprocess_params = {}
        if frame_sampling_rate is not None:
            preprocess_params['frame_sampling_rate'] = frame_sampling_rate
        if num_frames is not None:
            preprocess_params['num_frames'] = num_frames
        postprocess_params = {}
        if top_k is not None:
            postprocess_params['top_k'] = top_k
        return (preprocess_params, {}, postprocess_params)

    def __call__(self, videos: Union[str, List[str]], **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Assign labels to the video(s) passed as inputs.\n\n        Args:\n            videos (`str`, `List[str]`):\n                The pipeline handles three types of videos:\n\n                - A string containing a http link pointing to a video\n                - A string containing a local path to a video\n\n                The pipeline accepts either a single video or a batch of videos, which must then be passed as a string.\n                Videos in a batch must all be in the same format: all as http links or all as local paths.\n            top_k (`int`, *optional*, defaults to 5):\n                The number of top labels that will be returned by the pipeline. If the provided number is higher than\n                the number of labels available in the model configuration, it will default to the number of labels.\n            num_frames (`int`, *optional*, defaults to `self.model.config.num_frames`):\n                The number of frames sampled from the video to run the classification on. If not provided, will default\n                to the number of frames specified in the model configuration.\n            frame_sampling_rate (`int`, *optional*, defaults to 1):\n                The sampling rate used to select frames from the video. If not provided, will default to 1, i.e. every\n                frame will be used.\n\n        Return:\n            A dictionary or a list of dictionaries containing result. If the input is a single video, will return a\n            dictionary, if the input is a list of several videos, will return a list of dictionaries corresponding to\n            the videos.\n\n            The dictionaries contain the following keys:\n\n            - **label** (`str`) -- The label identified by the model.\n            - **score** (`int`) -- The score attributed by the model for that label.\n        '
        return super().__call__(videos, **kwargs)

    def preprocess(self, video, num_frames=None, frame_sampling_rate=1):
        if False:
            for i in range(10):
                print('nop')
        if num_frames is None:
            num_frames = self.model.config.num_frames
        if video.startswith('http://') or video.startswith('https://'):
            video = BytesIO(requests.get(video).content)
        videoreader = VideoReader(video)
        videoreader.seek(0)
        start_idx = 0
        end_idx = num_frames * frame_sampling_rate - 1
        indices = np.linspace(start_idx, end_idx, num=num_frames, dtype=np.int64)
        video = videoreader.get_batch(indices).asnumpy()
        video = list(video)
        model_inputs = self.image_processor(video, return_tensors=self.framework)
        return model_inputs

    def _forward(self, model_inputs):
        if False:
            print('Hello World!')
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=5):
        if False:
            return 10
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels
        if self.framework == 'pt':
            probs = model_outputs.logits.softmax(-1)[0]
            (scores, ids) = probs.topk(top_k)
        else:
            raise ValueError(f'Unsupported framework: {self.framework}')
        scores = scores.tolist()
        ids = ids.tolist()
        return [{'score': score, 'label': self.model.config.id2label[_id]} for (score, _id) in zip(scores, ids)]