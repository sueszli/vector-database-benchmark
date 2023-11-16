from typing import List, Union
from ..utils import is_torch_available
from .base import Pipeline
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING
    from ..models.speecht5.modeling_speecht5 import SpeechT5HifiGan
DEFAULT_VOCODER_ID = 'microsoft/speecht5_hifigan'

class TextToAudioPipeline(Pipeline):
    """
    Text-to-audio generation pipeline using any `AutoModelForTextToWaveform` or `AutoModelForTextToSpectrogram`. This
    pipeline generates an audio file from an input text and optional other conditional inputs.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> pipe = pipeline(model="suno/bark-small")
    >>> output = pipe("Hey it's HuggingFace on the phone!")

    >>> audio = output["audio"]
    >>> sampling_rate = output["sampling_rate"]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    <Tip>

    You can specify parameters passed to the model by using [`TextToAudioPipeline.__call__.forward_params`] or
    [`TextToAudioPipeline.__call__.generate_kwargs`].

    Example:

    ```python
    >>> from transformers import pipeline

    >>> music_generator = pipeline(task="text-to-audio", model="facebook/musicgen-small", framework="pt")

    >>> # diversify the music generation by adding randomness with a high temperature and set a maximum music length
    >>> generate_kwargs = {
    ...     "do_sample": True,
    ...     "temperature": 0.7,
    ...     "max_new_tokens": 35,
    ... }

    >>> outputs = music_generator("Techno music with high melodic riffs", generate_kwargs=generate_kwargs)
    ```

    </Tip>

    This pipeline can currently be loaded from [`pipeline`] using the following task identifiers: `"text-to-speech"` or
    `"text-to-audio"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=text-to-speech).
    """

    def __init__(self, *args, vocoder=None, sampling_rate=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        if self.framework == 'tf':
            raise ValueError('The TextToAudioPipeline is only available in PyTorch.')
        self.vocoder = None
        if self.model.__class__ in MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING.values():
            self.vocoder = SpeechT5HifiGan.from_pretrained(DEFAULT_VOCODER_ID).to(self.model.device) if vocoder is None else vocoder
        self.sampling_rate = sampling_rate
        if self.vocoder is not None:
            self.sampling_rate = self.vocoder.config.sampling_rate
        if self.sampling_rate is None:
            config = self.model.config
            gen_config = self.model.__dict__.get('generation_config', None)
            if gen_config is not None:
                config.update(gen_config.to_dict())
            for sampling_rate_name in ['sample_rate', 'sampling_rate']:
                sampling_rate = getattr(config, sampling_rate_name, None)
                if sampling_rate is not None:
                    self.sampling_rate = sampling_rate

    def preprocess(self, text, **kwargs):
        if False:
            i = 10
            return i + 15
        if isinstance(text, str):
            text = [text]
        if self.model.config.model_type == 'bark':
            new_kwargs = {'max_length': self.model.generation_config.semantic_config.get('max_input_semantic_length', 256), 'add_special_tokens': False, 'return_attention_mask': True, 'return_token_type_ids': False, 'padding': 'max_length'}
            new_kwargs.update(kwargs)
            kwargs = new_kwargs
        output = self.tokenizer(text, **kwargs, return_tensors='pt')
        return output

    def _forward(self, model_inputs, **kwargs):
        if False:
            print('Hello World!')
        kwargs = self._ensure_tensor_on_device(kwargs, device=self.device)
        forward_params = kwargs['forward_params']
        generate_kwargs = kwargs['generate_kwargs']
        if self.model.can_generate():
            generate_kwargs = self._ensure_tensor_on_device(generate_kwargs, device=self.device)
            forward_params.update(generate_kwargs)
            output = self.model.generate(**model_inputs, **forward_params)
        else:
            if len(generate_kwargs):
                raise ValueError(f"You're using the `TextToAudioPipeline` with a forward-only model, but `generate_kwargs` is non empty.\n                                 For forward-only TTA models, please use `forward_params` instead of of\n                                 `generate_kwargs`. For reference, here are the `generate_kwargs` used here:\n                                 {generate_kwargs.keys()}")
            output = self.model(**model_inputs, **forward_params)[0]
        if self.vocoder is not None:
            output = self.vocoder(output)
        return output

    def __call__(self, text_inputs: Union[str, List[str]], **forward_params):
        if False:
            return 10
        '\n        Generates speech/audio from the inputs. See the [`TextToAudioPipeline`] documentation for more information.\n\n        Args:\n            text_inputs (`str` or `List[str]`):\n                The text(s) to generate.\n            forward_params (`dict`, *optional*):\n                Parameters passed to the model generation/forward method. `forward_params` are always passed to the\n                underlying model.\n            generate_kwargs (`dict`, *optional*):\n                The dictionary of ad-hoc parametrization of `generate_config` to be used for the generation call. For a\n                complete overview of generate, check the [following\n                guide](https://huggingface.co/docs/transformers/en/main_classes/text_generation). `generate_kwargs` are\n                only passed to the underlying model if the latter is a generative model.\n\n        Return:\n            A `dict` or a list of `dict`: The dictionaries have two keys:\n\n            - **audio** (`np.ndarray` of shape `(nb_channels, audio_length)`) -- The generated audio waveform.\n            - **sampling_rate** (`int`) -- The sampling rate of the generated audio waveform.\n        '
        return super().__call__(text_inputs, **forward_params)

    def _sanitize_parameters(self, preprocess_params=None, forward_params=None, generate_kwargs=None):
        if False:
            while True:
                i = 10
        params = {'forward_params': forward_params if forward_params else {}, 'generate_kwargs': generate_kwargs if generate_kwargs else {}}
        if preprocess_params is None:
            preprocess_params = {}
        postprocess_params = {}
        return (preprocess_params, params, postprocess_params)

    def postprocess(self, waveform):
        if False:
            i = 10
            return i + 15
        output_dict = {}
        output_dict['audio'] = waveform.cpu().float().numpy()
        output_dict['sampling_rate'] = self.sampling_rate
        return output_dict