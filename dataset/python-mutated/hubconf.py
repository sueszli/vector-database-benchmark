dependencies = ['torch', 'gdown', 'pysbd', 'gruut', 'anyascii', 'pypinyin', 'coqpit', 'mecab-python3', 'unidic-lite']
import torch
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

def tts(model_name='tts_models/en/ljspeech/tacotron2-DCA', vocoder_name=None, use_cuda=False):
    if False:
        i = 10
        return i + 15
    'TTS entry point for PyTorch Hub that provides a Synthesizer object to synthesize speech from a give text.\n\n    Example:\n        >>> synthesizer = torch.hub.load(\'coqui-ai/TTS\', \'tts\', source=\'github\')\n        >>> wavs = synthesizer.tts("This is a test! This is also a test!!")\n            wavs - is a list of values of the synthesized speech.\n\n    Args:\n        model_name (str, optional): One of the model names from .model.json. Defaults to \'tts_models/en/ljspeech/tacotron2-DCA\'.\n        vocoder_name (str, optional): One of the model names from .model.json. Defaults to \'vocoder_models/en/ljspeech/multiband-melgan\'.\n        pretrained (bool, optional): [description]. Defaults to True.\n\n    Returns:\n        TTS.utils.synthesizer.Synthesizer: Synthesizer object wrapping both vocoder and tts models.\n    '
    manager = ModelManager()
    (model_path, config_path, model_item) = manager.download_model(model_name)
    vocoder_name = model_item['default_vocoder'] if vocoder_name is None else vocoder_name
    (vocoder_path, vocoder_config_path, _) = manager.download_model(vocoder_name)
    synt = Synthesizer(tts_checkpoint=model_path, tts_config_path=config_path, vocoder_checkpoint=vocoder_path, vocoder_config=vocoder_config_path, use_cuda=use_cuda)
    return synt
if __name__ == '__main__':
    synthesizer = torch.hub.load('coqui-ai/TTS:dev', 'tts', source='github')
    synthesizer.tts('This is a test!')