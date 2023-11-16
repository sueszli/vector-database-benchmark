import logging
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional
from fairseq.data import Dictionary
logger = logging.getLogger(__name__)

def get_config_from_yaml(yaml_path: Path):
    if False:
        while True:
            i = 10
    try:
        import yaml
    except ImportError:
        print('Please install PyYAML: pip install PyYAML')
    config = {}
    if yaml_path.is_file():
        try:
            with open(yaml_path) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            raise Exception(f'Failed to load config from {yaml_path.as_posix()}: {e}')
    else:
        raise FileNotFoundError(f'{yaml_path.as_posix()} not found')
    return config

class S2TDataConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path: Path):
        if False:
            print('Hello World!')
        self.config = get_config_from_yaml(yaml_path)
        self.root = yaml_path.parent

    def _auto_convert_to_abs_path(self, x):
        if False:
            return 10
        if isinstance(x, str):
            if not Path(x).exists() and (self.root / x).exists():
                return (self.root / x).as_posix()
        elif isinstance(x, dict):
            return {k: self._auto_convert_to_abs_path(v) for (k, v) in x.items()}
        return x

    @property
    def vocab_filename(self):
        if False:
            for i in range(10):
                print('nop')
        'fairseq vocabulary file under data root'
        return self.config.get('vocab_filename', 'dict.txt')

    @property
    def speaker_set_filename(self):
        if False:
            for i in range(10):
                print('nop')
        'speaker set file under data root'
        return self.config.get('speaker_set_filename', None)

    @property
    def shuffle(self) -> bool:
        if False:
            while True:
                i = 10
        'Shuffle dataset samples before batching'
        return self.config.get('shuffle', False)

    @property
    def pre_tokenizer(self) -> Dict:
        if False:
            print('Hello World!')
        'Pre-tokenizer to apply before subword tokenization. Returning\n        a dictionary with `tokenizer` providing the tokenizer name and\n        the other items providing the tokenizer-specific arguments.\n        Tokenizers are defined in `fairseq.data.encoders.*`'
        tokenizer = self.config.get('pre_tokenizer', {'tokenizer': None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def bpe_tokenizer(self) -> Dict:
        if False:
            return 10
        'Subword tokenizer to apply after pre-tokenization. Returning\n        a dictionary with `bpe` providing the tokenizer name and\n        the other items providing the tokenizer-specific arguments.\n        Tokenizers are defined in `fairseq.data.encoders.*`'
        tokenizer = self.config.get('bpe_tokenizer', {'bpe': None})
        return self._auto_convert_to_abs_path(tokenizer)

    @property
    def prepend_tgt_lang_tag(self) -> bool:
        if False:
            return 10
        'Prepend target lang ID token as the target BOS (e.g. for to-many\n        multilingual setting). During inference, this requires `--prefix-size 1`\n        to force BOS to be lang ID token.'
        return self.config.get('prepend_tgt_lang_tag', False)

    @property
    def prepend_bos_and_append_tgt_lang_tag(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Prepend BOS and append target lang ID token to the target (e.g. mBART with language token pretraining).'
        return self.config.get('prepend_bos_and_append_tgt_lang_tag', False)

    @property
    def input_feat_per_channel(self):
        if False:
            print('Hello World!')
        'The dimension of input features (per audio channel)'
        return self.config.get('input_feat_per_channel', 80)

    @property
    def input_channels(self):
        if False:
            while True:
                i = 10
        'The number of channels in the input audio'
        return self.config.get('input_channels', 1)

    @property
    def sample_rate(self):
        if False:
            return 10
        return self.config.get('sample_rate', 16000)

    @property
    def sampling_alpha(self):
        if False:
            i = 10
            return i + 15
        'Hyper-parameter alpha = 1/T for temperature-based resampling.\n        (alpha = 1 for no resampling)'
        return self.config.get('sampling_alpha', 1.0)

    @property
    def use_audio_input(self):
        if False:
            return 10
        'Needed by the dataset loader to see if the model requires\n        raw audio as inputs.'
        return self.config.get('use_audio_input', False)

    def standardize_audio(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.use_audio_input and self.config.get('standardize_audio', False)

    @property
    def use_sample_rate(self):
        if False:
            while True:
                i = 10
        'Needed by the dataset loader to see if the model requires\n        raw audio with specific sample rate as inputs.'
        return self.config.get('use_sample_rate', 16000)

    @property
    def audio_root(self):
        if False:
            while True:
                i = 10
        'Audio paths in the manifest TSV can be relative and this provides\n        the root path. Set this to empty string when using absolute paths.'
        return self.config.get('audio_root', '')

    def get_transforms(self, transform_type, split, is_train):
        if False:
            return 10
        'Split-specific feature transforms. Allowing train set\n        wildcard `_train`, evaluation set wildcard `_eval` and general\n        wildcard `*` for matching.'
        from copy import deepcopy
        cfg = deepcopy(self.config)
        _cur = cfg.get(f'{transform_type}transforms', {})
        cur = _cur.get(split)
        cur = _cur.get('_train') if cur is None and is_train else cur
        cur = _cur.get('_eval') if cur is None and (not is_train) else cur
        cur = _cur.get('*') if cur is None else cur
        return cur

    def get_feature_transforms(self, split, is_train):
        if False:
            print('Hello World!')
        cfg = deepcopy(self.config)
        cur = self.get_transforms('', split, is_train)
        if cur is not None:
            logger.warning('Auto converting transforms into feature_transforms, but transforms will be deprecated in the future. Please update this in the config.')
            ft_transforms = self.get_transforms('feature_', split, is_train)
            if ft_transforms:
                cur.extend(ft_transforms)
        else:
            cur = self.get_transforms('feature_', split, is_train)
        cfg['feature_transforms'] = cur
        return cfg

    def get_waveform_transforms(self, split, is_train):
        if False:
            return 10
        cfg = deepcopy(self.config)
        cfg['waveform_transforms'] = self.get_transforms('waveform_', split, is_train)
        return cfg

    def get_dataset_transforms(self, split, is_train):
        if False:
            print('Hello World!')
        cfg = deepcopy(self.config)
        cfg['dataset_transforms'] = self.get_transforms('dataset_', split, is_train)
        return cfg

    @property
    def global_cmvn_stats_npz(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        path = self.config.get('global_cmvn', {}).get('stats_npz_path', None)
        return self._auto_convert_to_abs_path(path)

    @property
    def vocoder(self) -> Dict[str, str]:
        if False:
            print('Hello World!')
        vocoder = self.config.get('vocoder', {'type': 'griffin_lim'})
        return self._auto_convert_to_abs_path(vocoder)

    @property
    def hub(self) -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        return self.config.get('hub', {})

class S2SDataConfig(S2TDataConfig):
    """Wrapper class for data config YAML"""

    @property
    def vocab_filename(self):
        if False:
            for i in range(10):
                print('nop')
        'fairseq vocabulary file under data root'
        return self.config.get('vocab_filename', None)

    @property
    def pre_tokenizer(self) -> Dict:
        if False:
            i = 10
            return i + 15
        return None

    @property
    def bpe_tokenizer(self) -> Dict:
        if False:
            return 10
        return None

    @property
    def input_transformed_channels(self):
        if False:
            print('Hello World!')
        'The number of channels in the audio after feature transforms'
        _cur = self.config.get('transforms', {})
        ft_transforms = self.config.get('feature_transforms', {})
        if _cur and ft_transforms:
            _cur.update(ft_transforms)
        else:
            _cur = self.config.get('feature_transforms', {})
        cur = _cur.get('_train', [])
        _channels = self.input_channels
        if 'delta_deltas' in cur:
            _channels *= 3
        return _channels

    @property
    def output_sample_rate(self):
        if False:
            return 10
        'The audio sample rate of output target speech'
        return self.config.get('output_sample_rate', 22050)

    @property
    def target_speaker_embed(self):
        if False:
            for i in range(10):
                print('nop')
        'Target speaker embedding file (one line per target audio sample)'
        return self.config.get('target_speaker_embed', None)

    @property
    def prepend_tgt_lang_tag_as_bos(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Prepend target lang ID token as the target BOS.'
        return self.config.get('prepend_tgt_lang_tag_as_bos', False)

class MultitaskConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path: Path):
        if False:
            while True:
                i = 10
        config = get_config_from_yaml(yaml_path)
        self.config = {}
        for (k, v) in config.items():
            self.config[k] = SingleTaskConfig(k, v)

    def get_all_tasks(self):
        if False:
            print('Hello World!')
        return self.config

    def get_single_task(self, name):
        if False:
            for i in range(10):
                print('nop')
        assert name in self.config, f"multitask '{name}' does not exist!"
        return self.config[name]

    @property
    def first_pass_decoder_task_index(self):
        if False:
            for i in range(10):
                print('nop')
        "Return the task index of the first-pass text decoder.\n        If there are multiple 'is_first_pass_decoder: True' in the config file,\n            the last task is used for the first-pass decoder.\n        If there is no 'is_first_pass_decoder: True' in the config file,\n            the last task whose task_name includes 'target' and decoder_type is not ctc.\n        "
        idx = -1
        for (i, (k, v)) in enumerate(self.config.items()):
            if v.is_first_pass_decoder:
                idx = i
        if idx < 0:
            for (i, (k, v)) in enumerate(self.config.items()):
                if k.startswith('target') and v.decoder_type == 'transformer':
                    idx = i
        return idx

class SingleTaskConfig(object):

    def __init__(self, name, config):
        if False:
            i = 10
            return i + 15
        self.task_name = name
        self.config = config
        dict_path = config.get('dict', '')
        self.tgt_dict = Dictionary.load(dict_path) if Path(dict_path).exists() else None

    @property
    def data(self):
        if False:
            for i in range(10):
                print('nop')
        return self.config.get('data', '')

    @property
    def decoder_type(self):
        if False:
            i = 10
            return i + 15
        return self.config.get('decoder_type', 'transformer')

    @property
    def decoder_args(self):
        if False:
            return 10
        'Decoder arch related args'
        args = self.config.get('decoder_args', {})
        return Namespace(**args)

    @property
    def criterion_cfg(self):
        if False:
            while True:
                i = 10
        'cfg for the multitask criterion'
        if self.decoder_type == 'ctc':
            from fairseq.criterions.ctc import CtcCriterionConfig
            cfg = CtcCriterionConfig
            cfg.zero_infinity = self.config.get('zero_infinity', True)
        else:
            from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterionConfig
            cfg = LabelSmoothedCrossEntropyCriterionConfig
            cfg.label_smoothing = self.config.get('label_smoothing', 0.2)
        return cfg

    @property
    def input_from(self):
        if False:
            while True:
                i = 10
        'Condition on encoder/decoder of the main model'
        return 'decoder' if 'decoder_layer' in self.config else 'encoder'

    @property
    def input_layer(self):
        if False:
            i = 10
            return i + 15
        if self.input_from == 'decoder':
            return self.config['decoder_layer'] - 1
        else:
            return self.config.get('encoder_layer', 0) - 1

    @property
    def loss_weight_schedule(self):
        if False:
            i = 10
            return i + 15
        return 'decay' if 'loss_weight_max' in self.config and 'loss_weight_decay_steps' in self.config else 'fixed'

    def get_loss_weight(self, num_updates):
        if False:
            for i in range(10):
                print('nop')
        if self.loss_weight_schedule == 'fixed':
            weight = self.config.get('loss_weight', 1.0)
        else:
            assert self.config.get('loss_weight_decay_steps', 0) > 0, 'loss_weight_decay_steps must be greater than 0 for a decay schedule'
            loss_weight_min = self.config.get('loss_weight_min', 0.0001)
            loss_weight_decay_stepsize = (self.config['loss_weight_max'] - loss_weight_min) / self.config['loss_weight_decay_steps']
            weight = max(self.config['loss_weight_max'] - loss_weight_decay_stepsize * num_updates, loss_weight_min)
        return weight

    @property
    def prepend_bos_and_append_tgt_lang_tag(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Prepend BOS and append target lang ID token to the target (e.g. mBART with language token pretraining).'
        return self.config.get('prepend_bos_and_append_tgt_lang_tag', False)

    @property
    def eos_token(self):
        if False:
            i = 10
            return i + 15
        'EOS token during generation'
        return self.config.get('eos_token', '<eos>')

    @property
    def rdrop_alpha(self):
        if False:
            while True:
                i = 10
        return self.config.get('rdrop_alpha', 0.0)

    @property
    def is_first_pass_decoder(self):
        if False:
            while True:
                i = 10
        flag = self.config.get('is_first_pass_decoder', False)
        if flag:
            if self.decoder_type == 'ctc':
                raise ValueError('First-pass decoder in the multi-decoder model must not be CTC.')
            if 'target' not in self.task_name:
                raise Warning('The name of the first-pass decoder does not include "target".')
        return flag

    @property
    def get_lang_tag_mapping(self):
        if False:
            i = 10
            return i + 15
        return self.config.get('lang_tag_mapping', {})