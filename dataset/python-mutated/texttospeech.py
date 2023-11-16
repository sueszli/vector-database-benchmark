"""
TextToSpeech module
"""
try:
    import onnxruntime as ort
    from ttstokenizer import TTSTokenizer
    TTS = True
except ImportError:
    TTS = False
import torch
import yaml
import numpy as np
from huggingface_hub import hf_hub_download
from ..base import Pipeline

class TextToSpeech(Pipeline):
    """
    Generates speech from text
    """

    def __init__(self, path=None, maxtokens=512):
        if False:
            i = 10
            return i + 15
        '\n        Creates a new TextToSpeech pipeline.\n\n        Args:\n            path: optional Hugging Face model hub id\n            maxtokens: maximum number of tokens model can process, defaults to 512\n        '
        if not TTS:
            raise ImportError('TextToSpeech pipeline is not available - install "pipeline" extra to enable')
        path = path if path else 'neuml/ljspeech-jets-onnx'
        config = hf_hub_download(path, filename='config.yaml')
        model = hf_hub_download(path, filename='model.onnx')
        with open(config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        tokens = config.get('token', {}).get('list')
        self.tokenizer = TTSTokenizer(tokens)
        self.model = ort.InferenceSession(model, ort.SessionOptions(), self.providers())
        self.maxtokens = maxtokens
        self.input = self.model.get_inputs()[0].name

    def __call__(self, text):
        if False:
            i = 10
            return i + 15
        '\n        Generates speech from text. Text longer than maxtokens will be batched and returned\n        as a single waveform per text input.\n\n        This method supports files as a string or a list. If the input is a string,\n        the return type is string. If text is a list, the return type is a list.\n\n        Args:\n            text: text|list\n\n        Returns:\n            list of speech as NumPy array waveforms\n        '
        texts = [text] if isinstance(text, str) else text
        outputs = []
        for x in texts:
            x = self.tokenizer(x)
            result = self.execute(x)
            outputs.append(result)
        return outputs[0] if isinstance(text, str) else outputs

    def providers(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a list of available and usable providers.\n\n        Returns:\n            list of available and usable providers\n        '
        if torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
            return [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'}), 'CPUExecutionProvider']
        return ['CPUExecutionProvider']

    def execute(self, tokens):
        if False:
            while True:
                i = 10
        '\n        Executes model run for input array of tokens. This method will build batches\n        of tokens when len(tokens) > maxtokens.\n\n        Args:\n            tokens: array of tokens to pass to model\n\n        Returns:\n            waveform as NumPy array\n        '
        results = []
        for x in self.batch(tokens, self.maxtokens):
            output = self.model.run(None, {self.input: x})
            results.append(output[0])
        return np.concatenate(results)