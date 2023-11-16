from typing import Any, Dict, Union
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import WavToScp
from modelscope.utils.audio.audio_utils import extract_pcm_from_wav, load_bytes_from_url
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()
__all__ = ['WeNetAutomaticSpeechRecognitionPipeline']

@PIPELINES.register_module(Tasks.auto_speech_recognition, module_name=Pipelines.asr_wenet_inference)
class WeNetAutomaticSpeechRecognitionPipeline(Pipeline):
    """ASR Inference Pipeline
    """

    def __init__(self, model: Union[Model, str]=None, preprocessor: WavToScp=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'use `model` and `preprocessor` to create an asr pipeline for prediction\n        '
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def __call__(self, audio_in: Union[str, bytes], audio_fs: int=None, recog_type: str=None, audio_format: str=None) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        from funasr.utils import asr_utils
        self.recog_type = recog_type
        self.audio_format = audio_format
        self.audio_fs = audio_fs
        if isinstance(audio_in, str):
            (self.audio_in, checking_audio_fs) = load_bytes_from_url(audio_in)
        elif isinstance(audio_in, bytes):
            (self.audio_in, checking_audio_fs) = extract_pcm_from_wav(audio_in)
        else:
            self.audio_in = audio_in
        if checking_audio_fs is not None:
            self.audio_fs = checking_audio_fs
        if recog_type is None or audio_format is None:
            (self.recog_type, self.audio_format, self.audio_in) = asr_utils.type_checking(audio_in=self.audio_in, recog_type=recog_type, audio_format=audio_format)
        if hasattr(asr_utils, 'sample_rate_checking'):
            checking_audio_fs = asr_utils.sample_rate_checking(self.audio_in, self.audio_format)
            if checking_audio_fs is not None:
                self.audio_fs = checking_audio_fs
        inputs = {'audio': self.audio_in, 'audio_format': self.audio_format, 'audio_fs': self.audio_fs}
        output = self.forward(inputs)
        rst = self.postprocess(output['asr_result'])
        return rst

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        'Decoding\n        '
        inputs['asr_result'] = self.model(inputs)
        return inputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'process the asr results\n        '
        return inputs