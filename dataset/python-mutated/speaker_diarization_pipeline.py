import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import numpy
import yaml
from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.audio.audio_utils import generate_scp_for_sv, generate_sd_scp_from_url, update_local_model
from modelscope.utils.constant import Frameworks, ModelFile, Tasks
from modelscope.utils.hub import snapshot_download
from modelscope.utils.logger import get_logger
logger = get_logger()
__all__ = ['SpeakerDiarizationPipeline']

@PIPELINES.register_module(Tasks.speaker_diarization, module_name=Pipelines.speaker_diarization_inference)
class SpeakerDiarizationPipeline(Pipeline):
    """Speaker Diarization Inference Pipeline
    use `model` to create a Speaker Diarization pipeline.

    Args:
        model (SpeakerDiarizationPipeline): A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the preprocessor's constructor.
    Examples:
        >>> from modelscope.pipelines import pipeline
        >>> pipeline_sd = pipeline(
        >>>    task=Tasks.speaker_diarization, model='damo/xxxxxxxxxxxxx')
        >>> audio_in=('','','','')
        >>> print(pipeline_sd(audio_in))

    """

    def __init__(self, model: Union[Model, str]=None, sv_model: Optional[Union[Model, str]]=None, sv_model_revision: Optional[str]=None, ngpu: int=1, **kwargs):
        if False:
            return 10
        "use `model` to create a speaker diarization pipeline for prediction\n        Args:\n            model ('Model' or 'str'):\n                The pipeline handles three types of model:\n\n                - A model instance\n                - A model local dir\n                - A model id in the model hub\n            sv_model (Optional: 'Model' or 'str'):\n                speaker verification model from model hub or local\n                example: 'damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'\n            sv_model_revision (Optional: 'str'):\n                speaker verfication model revision from model hub\n        "
        super().__init__(model=model, **kwargs)
        self.model_cfg = None
        config_path = os.path.join(model, ModelFile.CONFIGURATION)
        self.sv_model = sv_model
        self.sv_model_revision = sv_model_revision
        self.cmd = self.get_cmd(config_path, kwargs, model)
        from funasr.bin import diar_inference_launch
        self.funasr_infer_modelscope = diar_inference_launch.inference_launch(mode=self.cmd['mode'], output_dir=self.cmd['output_dir'], batch_size=self.cmd['batch_size'], dtype=self.cmd['dtype'], ngpu=ngpu, seed=self.cmd['seed'], num_workers=self.cmd['num_workers'], log_level=self.cmd['log_level'], key_file=self.cmd['key_file'], diar_train_config=self.cmd['diar_train_config'], diar_model_file=self.cmd['diar_model_file'], model_tag=self.cmd['model_tag'], allow_variable_data_keys=self.cmd['allow_variable_data_keys'], streaming=self.cmd['streaming'], smooth_size=self.cmd['smooth_size'], dur_threshold=self.cmd['dur_threshold'], out_format=self.cmd['out_format'], param_dict=self.cmd['param_dict'], **kwargs)

    def __call__(self, audio_in: Union[tuple, str, Any]=None, output_dir: str=None, param_dict: dict=None) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Decoding the input audios\n        Args:\n            audio_in('str' or 'bytes'):\n                - A string containing a local path to a wav file\n                - A string containing a local path to a scp\n                - A string containing a wav url\n                - A bytes input\n            output_dir('str'):\n                output dir\n            param_dict('dict'):\n                extra kwargs\n        Return:\n            A dictionary of result or a list of dictionary of result.\n\n            The dictionary contain the following keys:\n            - **text** ('str') --The speaker diarization result.\n        "
        if len(audio_in) == 0:
            raise ValueError('The input of sv should not be null.')
        else:
            self.audio_in = audio_in
        if output_dir is not None:
            self.cmd['output_dir'] = output_dir
        self.cmd['param_dict'] = param_dict
        output = self.forward(self.audio_in)
        result = self.postprocess(output)
        return result

    def postprocess(self, inputs: list) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Postprocessing\n        '
        rst = {}
        for i in range(len(inputs)):
            if i == 0 and len(inputs) == 1:
                rst[OutputKeys.TEXT] = inputs[0]['value']
            else:
                rst[inputs[i]['key']] = inputs[i]['value']
        return rst

    def get_cmd(self, config_path, extra_args, model_path) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        self.model_cfg = json.loads(open(config_path).read())
        model_dir = os.path.dirname(config_path)
        mode = self.model_cfg['model']['model_config']['mode']
        diar_model_path = os.path.join(model_dir, self.model_cfg['model']['model_config']['diar_model_name'])
        diar_model_config = os.path.join(model_dir, self.model_cfg['model']['model_config']['diar_model_config'])
        update_local_model(self.model_cfg['model']['model_config'], model_path, extra_args)
        cmd = {'mode': mode, 'output_dir': None, 'batch_size': 1, 'dtype': 'float32', 'ngpu': 1, 'seed': 0, 'num_workers': 0, 'log_level': 'ERROR', 'key_file': None, 'diar_model_file': diar_model_path, 'diar_train_config': diar_model_config, 'model_tag': None, 'allow_variable_data_keys': True, 'streaming': False, 'smooth_size': 83, 'dur_threshold': 10, 'out_format': 'vad', 'param_dict': {'sv_model_file': None, 'sv_train_config': None}}
        user_args_dict = ['mode', 'output_dir', 'batch_size', 'ngpu', 'log_level', 'allow_variable_data_keys', 'streaming', 'num_workers', 'smooth_size', 'dur_threshold', 'out_format', 'param_dict']
        model_config = self.model_cfg['model']['model_config']
        if model_config.__contains__('sv_model') and self.sv_model != '':
            self.sv_model = model_config['sv_model']
        if model_config.__contains__('sv_model_revision'):
            self.sv_model_revision = model_config['sv_model_revision']
        self.load_sv_model(cmd)
        for user_args in user_args_dict:
            if user_args in extra_args:
                if extra_args.get(user_args) is not None:
                    if isinstance(cmd[user_args], dict) and isinstance(extra_args[user_args], dict):
                        cmd[user_args].update(extra_args[user_args])
                    else:
                        cmd[user_args] = extra_args[user_args]
                del extra_args[user_args]
        return cmd

    def load_sv_model(self, cmd):
        if False:
            print('Hello World!')
        if self.sv_model is not None and self.sv_model != '':
            if os.path.exists(self.sv_model):
                sv_model = self.sv_model
            else:
                sv_model = snapshot_download(self.sv_model, revision=self.sv_model_revision)
            logger.info('loading speaker verification model from {0} ...'.format(sv_model))
            config_path = os.path.join(sv_model, ModelFile.CONFIGURATION)
            model_cfg = json.loads(open(config_path).read())
            model_dir = os.path.dirname(config_path)
            cmd['param_dict']['sv_model_file'] = os.path.join(model_dir, model_cfg['model']['model_config']['sv_model_name'])
            cmd['param_dict']['sv_train_config'] = os.path.join(model_dir, model_cfg['model']['model_config']['sv_model_config'])

    def forward(self, audio_in: Union[tuple, str, Any]=None) -> list:
        if False:
            i = 10
            return i + 15
        'Decoding\n        '
        if isinstance(audio_in, str) or (isinstance(audio_in, tuple) and all((isinstance(item, str) for item in audio_in))):
            logger.info(f'Speaker Verification Processing: {audio_in} ...')
        else:
            logger.info(f'Speaker Verification Processing: {str(audio_in)[:100]} ...')
        (data_cmd, raw_inputs) = (None, None)
        if isinstance(audio_in, tuple) or isinstance(audio_in, list):
            if isinstance(audio_in[0], str):
                if len(audio_in[0].split(',')) == 3 and audio_in[0].split(',')[0].endswith('.scp'):
                    data_cmd = []
                    for audio_cmd in audio_in:
                        if len(audio_cmd.split(',')) == 3 and audio_cmd.split(',')[0].endswith('.scp'):
                            data_cmd.append(tuple(audio_cmd.split(',')))
                else:
                    raw_inputs = generate_sd_scp_from_url(audio_in)
            elif isinstance(audio_in[0], (bytes, numpy.ndarray)):
                raw_inputs = audio_in
            else:
                raise TypeError('Unsupported data type, it must be data_name_type_path, file_path, url, bytes or numpy.ndarray')
        else:
            raise TypeError('audio_in must be a list of data_name_type_path, file_path, url, bytes or numpy.ndarray')
        self.cmd['name_and_type'] = data_cmd
        self.cmd['raw_inputs'] = raw_inputs
        result = self.run_inference(self.cmd)
        return result

    def run_inference(self, cmd):
        if False:
            return 10
        if self.framework == Frameworks.torch:
            diar_result = self.funasr_infer_modelscope(data_path_and_name_and_type=cmd['name_and_type'], raw_inputs=cmd['raw_inputs'], output_dir_v2=cmd['output_dir'], param_dict=cmd['param_dict'])
        else:
            raise ValueError('framework is mismatching, which should be pytorch')
        return diar_result