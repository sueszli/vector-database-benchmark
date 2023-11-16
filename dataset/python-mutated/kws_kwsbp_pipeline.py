import os
from typing import Any, Dict, List, Union
import json
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import WavToLists
from modelscope.utils.audio.audio_utils import extract_pcm_from_wav, load_bytes_from_url
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()
__all__ = ['KeyWordSpottingKwsbpPipeline']

@PIPELINES.register_module(Tasks.keyword_spotting, module_name=Pipelines.kws_kwsbp)
class KeyWordSpottingKwsbpPipeline(Pipeline):
    """KWS Pipeline - key word spotting decoding
    """

    def __init__(self, model: Union[Model, str]=None, preprocessor: WavToLists=None, **kwargs):
        if False:
            while True:
                i = 10
        'use `model` and `preprocessor` to create a kws pipeline for prediction\n        '
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def __call__(self, audio_in: Union[List[str], str, bytes], **kwargs) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        if 'keywords' in kwargs.keys():
            self.keywords = kwargs['keywords']
            if isinstance(self.keywords, str):
                word_list = []
                word = {}
                word['keyword'] = self.keywords
                word_list.append(word)
                self.keywords = word_list
        else:
            self.keywords = None
        if self.preprocessor is None:
            self.preprocessor = WavToLists()
        if isinstance(audio_in, str):
            (audio_in, audio_fs) = load_bytes_from_url(audio_in)
        elif isinstance(audio_in, bytes):
            (audio_in, audio_fs) = extract_pcm_from_wav(audio_in)
        output = self.preprocessor.forward(self.model.forward(None), audio_in)
        output = self.forward(output)
        rst = self.postprocess(output)
        return rst

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Decoding\n        '
        logger.info(f"Decoding with {inputs['kws_type']} mode ...")
        out = self.run_with_kwsbp(inputs)
        return out

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        "process the kws results\n\n        Args:\n          inputs['pos_kws_list'] or inputs['neg_kws_list']:\n          result_dict format example:\n            [{\n              'confidence': 0.9903678297996521,\n              'filename': 'data/test/audios/kws_xiaoyunxiaoyun.wav',\n              'keyword': '小云小云',\n              'offset': 5.760000228881836,  # second\n              'rtf_time': 66,               # millisecond\n              'threshold': 0,\n              'wav_time': 9.1329375         # second\n            }]\n        "
        import kws_util.common
        neg_kws_list = None
        pos_kws_list = None
        if 'pos_kws_list' in inputs:
            pos_kws_list = inputs['pos_kws_list']
        if 'neg_kws_list' in inputs:
            neg_kws_list = inputs['neg_kws_list']
        rst_dict = kws_util.common.parsing_kws_result(kws_type=inputs['kws_type'], pos_list=pos_kws_list, neg_list=neg_kws_list)
        if 'kws_list' not in rst_dict:
            rst_dict['kws_list'] = []
        return rst_dict

    def run_with_kwsbp(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        import kwsbp
        import kws_util.common
        kws_inference = kwsbp.KwsbpEngine()
        cmd = {'sys_dir': inputs['model_workspace'], 'cfg_file': inputs['cfg_file_path'], 'sample_rate': inputs['sample_rate'], 'keyword_custom': '', 'pcm_data': None, 'pcm_data_len': 0, 'list_flag': True, 'customized_keywords': kws_util.common.generate_customized_keywords(self.keywords)}
        if inputs['kws_type'] == 'pcm':
            cmd['pcm_data'] = inputs['pos_data']
            cmd['pcm_data_len'] = len(inputs['pos_data'])
            cmd['list_flag'] = False
        if inputs['kws_type'] == 'roc':
            inputs['keyword_grammar_path'] = os.path.join(inputs['model_workspace'], 'keywords_roc.json')
        if inputs['kws_type'] in ['wav', 'pcm', 'pos_testsets', 'roc']:
            cmd['wave_scp'] = inputs['pos_wav_list']
            cmd['keyword_grammar_path'] = inputs['keyword_grammar_path']
            cmd['num_thread'] = inputs['pos_num_thread']
            if hasattr(kws_inference, 'inference_new'):
                result = kws_inference.inference_new(cmd['sys_dir'], cmd['cfg_file'], cmd['keyword_grammar_path'], str(json.dumps(cmd['wave_scp'])), str(cmd['customized_keywords']), cmd['pcm_data'], cmd['pcm_data_len'], cmd['sample_rate'], cmd['num_thread'], cmd['list_flag'])
            else:
                result = kws_inference.inference(cmd['sys_dir'], cmd['cfg_file'], cmd['keyword_grammar_path'], str(json.dumps(cmd['wave_scp'])), str(cmd['customized_keywords']), cmd['sample_rate'], cmd['num_thread'])
            pos_result = json.loads(result)
            inputs['pos_kws_list'] = pos_result['kws_list']
        if inputs['kws_type'] in ['neg_testsets', 'roc']:
            cmd['wave_scp'] = inputs['neg_wav_list']
            cmd['keyword_grammar_path'] = inputs['keyword_grammar_path']
            cmd['num_thread'] = inputs['neg_num_thread']
            if hasattr(kws_inference, 'inference_new'):
                result = kws_inference.inference_new(cmd['sys_dir'], cmd['cfg_file'], cmd['keyword_grammar_path'], str(json.dumps(cmd['wave_scp'])), str(cmd['customized_keywords']), cmd['pcm_data'], cmd['pcm_data_len'], cmd['sample_rate'], cmd['num_thread'], cmd['list_flag'])
            else:
                result = kws_inference.inference(cmd['sys_dir'], cmd['cfg_file'], cmd['keyword_grammar_path'], str(json.dumps(cmd['wave_scp'])), str(cmd['customized_keywords']), cmd['sample_rate'], cmd['num_thread'])
            neg_result = json.loads(result)
            inputs['neg_kws_list'] = neg_result['kws_list']
        return inputs