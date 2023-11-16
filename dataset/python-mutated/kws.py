import os
from typing import Any, Dict, List, Union
import yaml
from modelscope.metainfo import Preprocessors
from modelscope.models.base import Model
from modelscope.utils.constant import Fields
from .base import Preprocessor
from .builder import PREPROCESSORS
__all__ = ['WavToLists']

@PREPROCESSORS.register_module(Fields.audio, module_name=Preprocessors.wav_to_lists)
class WavToLists(Preprocessor):
    """generate audio lists file from wav
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def __call__(self, model: Model, audio_in: Union[List[str], str, bytes]) -> Dict[str, Any]:
        if False:
            return 10
        'Call functions to load model and wav.\n\n        Args:\n            model (Model): model should be provided\n            audio_in (Union[List[str], str, bytes]):\n                audio_in[0] is positive wav path, audio_in[1] is negative wav path;\n                audio_in (str) is positive wav path;\n                audio_in (bytes) is audio pcm data;\n        Returns:\n            Dict[str, Any]: the kws result\n        '
        self.model = model
        out = self.forward(self.model.forward(), audio_in)
        return out

    def forward(self, model: Dict[str, Any], audio_in: Union[List[str], str, bytes]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        assert len(model['config_path']) > 0, 'preprocess model[config_path] is empty'
        assert os.path.exists(model['config_path']), 'model config.yaml is absent'
        inputs = model.copy()
        import kws_util.common
        kws_type = kws_util.common.type_checking(audio_in)
        assert kws_type in ['wav', 'pcm', 'pos_testsets', 'neg_testsets', 'roc'], f'kws_type {kws_type} is invalid, please check audio data'
        inputs['kws_type'] = kws_type
        if kws_type == 'wav':
            inputs['pos_wav_path'] = audio_in
        elif kws_type == 'pcm':
            inputs['pos_data'] = audio_in
        if kws_type in ['pos_testsets', 'roc']:
            inputs['pos_wav_path'] = audio_in[0]
        if kws_type in ['neg_testsets', 'roc']:
            inputs['neg_wav_path'] = audio_in[1]
        out = self.read_config(inputs)
        out = self.generate_wav_lists(out)
        return out

    def read_config(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            return 10
        'read and parse config.yaml to get all model files\n        '
        assert os.path.exists(inputs['config_path']), 'model config yaml file does not exist'
        config_file = open(inputs['config_path'], encoding='utf-8')
        root = yaml.full_load(config_file)
        config_file.close()
        inputs['cfg_file'] = root['cfg_file']
        inputs['cfg_file_path'] = os.path.join(inputs['model_workspace'], root['cfg_file'])
        inputs['keyword_grammar'] = root['keyword_grammar']
        inputs['keyword_grammar_path'] = os.path.join(inputs['model_workspace'], root['keyword_grammar'])
        inputs['sample_rate'] = root['sample_rate']
        return inputs

    def generate_wav_lists(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'assemble wav lists\n        '
        import kws_util.common
        if inputs['kws_type'] == 'wav':
            wav_list = []
            wave_scp_content: str = inputs['pos_wav_path']
            wav_list.append(wave_scp_content)
            inputs['pos_wav_list'] = wav_list
            inputs['pos_wav_count'] = 1
            inputs['pos_num_thread'] = 1
        if inputs['kws_type'] == 'pcm':
            inputs['pos_wav_list'] = ['pcm_data']
            inputs['pos_wav_count'] = 1
            inputs['pos_num_thread'] = 1
        if inputs['kws_type'] in ['pos_testsets', 'roc']:
            wav_list = []
            wav_dir = inputs['pos_wav_path']
            wav_list = kws_util.common.recursion_dir_all_wav(wav_list, wav_dir)
            inputs['pos_wav_list'] = wav_list
            list_count: int = len(wav_list)
            inputs['pos_wav_count'] = list_count
            if list_count <= 128:
                inputs['pos_num_thread'] = list_count
            else:
                inputs['pos_num_thread'] = 128
        if inputs['kws_type'] in ['neg_testsets', 'roc']:
            wav_list = []
            wav_dir = inputs['neg_wav_path']
            wav_list = kws_util.common.recursion_dir_all_wav(wav_list, wav_dir)
            inputs['neg_wav_list'] = wav_list
            list_count: int = len(wav_list)
            inputs['neg_wav_count'] = list_count
            if list_count <= 128:
                inputs['neg_num_thread'] = list_count
            else:
                inputs['neg_num_thread'] = 128
        return inputs