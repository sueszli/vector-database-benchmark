from collections import UserDict
from typing import Union
import numpy as np
import requests
from ..utils import add_end_docstrings, logging
from .audio_classification import ffmpeg_read
from .base import PIPELINE_INIT_ARGS, Pipeline
logger = logging.get_logger(__name__)

@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotAudioClassificationPipeline(Pipeline):
    """
    Zero shot audio classification pipeline using `ClapModel`. This pipeline predicts the class of an audio when you
    provide an audio and a set of `candidate_labels`.

    Example:
    ```python
    >>> from transformers import pipeline
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("ashraq/esc50")
    >>> audio = next(iter(dataset["train"]["audio"]))["array"]
    >>> classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
    >>> classifier(audio, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])
    [{'score': 0.9996, 'label': 'Sound of a dog'}, {'score': 0.0004, 'label': 'Sound of vaccum cleaner'}]
    ```


    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial) This audio
    classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-audio-classification"`. See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-audio-classification).
    """

    def __init__(self, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        if self.framework != 'pt':
            raise ValueError(f'The {self.__class__} is only available in PyTorch.')

    def __call__(self, audios: Union[np.ndarray, bytes, str], **kwargs):
        if False:
            print('Hello World!')
        '\n        Assign labels to the audio(s) passed as inputs.\n\n        Args:\n            audios (`str`, `List[str]`, `np.array` or `List[np.array]`):\n                The pipeline handles three types of inputs:\n                - A string containing a http link pointing to an audio\n                - A string containing a local path to an audio\n                - An audio loaded in numpy\n            candidate_labels (`List[str]`):\n                The candidate labels for this audio\n            hypothesis_template (`str`, *optional*, defaults to `"This is a sound of {}"`):\n                The sentence used in cunjunction with *candidate_labels* to attempt the audio classification by\n                replacing the placeholder with the candidate_labels. Then likelihood is estimated by using\n                logits_per_audio\n        Return:\n            A list of dictionaries containing result, one dictionary per proposed label. The dictionaries contain the\n            following keys:\n            - **label** (`str`) -- The label identified by the model. It is one of the suggested `candidate_label`.\n            - **score** (`float`) -- The score attributed by the model for that label (between 0 and 1).\n        '
        return super().__call__(audios, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        preprocess_params = {}
        if 'candidate_labels' in kwargs:
            preprocess_params['candidate_labels'] = kwargs['candidate_labels']
        if 'hypothesis_template' in kwargs:
            preprocess_params['hypothesis_template'] = kwargs['hypothesis_template']
        return (preprocess_params, {}, {})

    def preprocess(self, audio, candidate_labels=None, hypothesis_template='This is a sound of {}.'):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(audio, str):
            if audio.startswith('http://') or audio.startswith('https://'):
                audio = requests.get(audio).content
            else:
                with open(audio, 'rb') as f:
                    audio = f.read()
        if isinstance(audio, bytes):
            audio = ffmpeg_read(audio, self.feature_extractor.sampling_rate)
        if not isinstance(audio, np.ndarray):
            raise ValueError('We expect a numpy ndarray as input')
        if len(audio.shape) != 1:
            raise ValueError('We expect a single channel audio input for ZeroShotAudioClassificationPipeline')
        inputs = self.feature_extractor([audio], sampling_rate=self.feature_extractor.sampling_rate, return_tensors='pt')
        inputs['candidate_labels'] = candidate_labels
        sequences = [hypothesis_template.format(x) for x in candidate_labels]
        text_inputs = self.tokenizer(sequences, return_tensors=self.framework, padding=True)
        inputs['text_inputs'] = [text_inputs]
        return inputs

    def _forward(self, model_inputs):
        if False:
            for i in range(10):
                print('nop')
        candidate_labels = model_inputs.pop('candidate_labels')
        text_inputs = model_inputs.pop('text_inputs')
        if isinstance(text_inputs[0], UserDict):
            text_inputs = text_inputs[0]
        else:
            text_inputs = text_inputs[0][0]
        outputs = self.model(**text_inputs, **model_inputs)
        model_outputs = {'candidate_labels': candidate_labels, 'logits': outputs.logits_per_audio}
        return model_outputs

    def postprocess(self, model_outputs):
        if False:
            while True:
                i = 10
        candidate_labels = model_outputs.pop('candidate_labels')
        logits = model_outputs['logits'][0]
        if self.framework == 'pt':
            probs = logits.softmax(dim=0)
            scores = probs.tolist()
        else:
            raise ValueError('`tf` framework not supported.')
        result = [{'score': score, 'label': candidate_label} for (score, candidate_label) in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])]
        return result