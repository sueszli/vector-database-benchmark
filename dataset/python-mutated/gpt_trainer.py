from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torchaudio
from coqpit import Coqpit
from torch.nn import functional as F
from torch.utils.data import DataLoader
from trainer.torch import DistributedSampler
from trainer.trainer_utils import get_optimizer, get_scheduler
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
from TTS.utils.io import load_fsspec

@dataclass
class GPTTrainerConfig(XttsConfig):
    lr: float = 5e-06
    training_seed: int = 1
    optimizer_wd_only_on_weights: bool = False
    weighted_loss_attrs: dict = field(default_factory=lambda : {})
    weighted_loss_multipliers: dict = field(default_factory=lambda : {})
    test_sentences: List[dict] = field(default_factory=lambda : [])

@dataclass
class XttsAudioConfig(XttsAudioConfig):
    dvae_sample_rate: int = 22050

@dataclass
class GPTArgs(XttsArgs):
    min_conditioning_length: int = 66150
    max_conditioning_length: int = 132300
    gpt_loss_text_ce_weight: float = 0.01
    gpt_loss_mel_ce_weight: float = 1.0
    gpt_num_audio_tokens: int = 8194
    debug_loading_failures: bool = False
    max_wav_length: int = 255995
    max_text_length: int = 200
    tokenizer_file: str = ''
    mel_norm_file: str = 'https://coqui.gateway.scarf.sh/v0.14.0_models/mel_norms.pth'
    dvae_checkpoint: str = ''
    xtts_checkpoint: str = ''
    gpt_checkpoint: str = ''
    vocoder: str = ''

def callback_clearml_load_save(operation_type, model_info):
    if False:
        print('Hello World!')
    assert operation_type in ('load', 'save')
    if 'similarities.pth' in model_info.__dict__['local_model_path']:
        return None
    return model_info

class GPTTrainer(BaseTTS):

    def __init__(self, config: Coqpit):
        if False:
            i = 10
            return i + 15
        '\n        Tortoise GPT training class\n        '
        super().__init__(config, ap=None, tokenizer=None)
        self.config = config
        self.xtts = Xtts(self.config)
        self.xtts.tokenizer = VoiceBpeTokenizer(self.args.tokenizer_file)
        self.xtts.init_models()
        if self.args.xtts_checkpoint:
            self.load_checkpoint(self.config, self.args.xtts_checkpoint, eval=False, strict=False)
        if self.args.mel_norm_file:
            self.xtts.mel_stats = load_fsspec(self.args.mel_norm_file)
        if self.args.gpt_checkpoint:
            gpt_checkpoint = torch.load(self.args.gpt_checkpoint, map_location=torch.device('cpu'))
            if 'model' in gpt_checkpoint.keys() and 'config' in gpt_checkpoint.keys():
                print('Coqui Trainer checkpoint detected! Converting it!')
                gpt_checkpoint = gpt_checkpoint['model']
                states_keys = list(gpt_checkpoint.keys())
                for key in states_keys:
                    if 'gpt.' in key:
                        new_key = key.replace('gpt.', '')
                        gpt_checkpoint[new_key] = gpt_checkpoint[key]
                        del gpt_checkpoint[key]
                    else:
                        del gpt_checkpoint[key]
            if 'text_embedding.weight' in gpt_checkpoint and gpt_checkpoint['text_embedding.weight'].shape != self.xtts.gpt.text_embedding.weight.shape:
                num_new_tokens = self.xtts.gpt.text_embedding.weight.shape[0] - gpt_checkpoint['text_embedding.weight'].shape[0]
                print(f' > Loading checkpoint with {num_new_tokens} additional tokens.')
                emb_g = gpt_checkpoint['text_embedding.weight']
                new_row = torch.randn(num_new_tokens, emb_g.shape[1])
                start_token_row = emb_g[-1, :]
                emb_g = torch.cat([emb_g, new_row], axis=0)
                emb_g[-1, :] = start_token_row
                gpt_checkpoint['text_embedding.weight'] = emb_g
                text_head_weight = gpt_checkpoint['text_head.weight']
                start_token_row = text_head_weight[-1, :]
                new_entry = torch.randn(num_new_tokens, self.xtts.gpt.text_head.weight.shape[1])
                text_head_weight = torch.cat([text_head_weight, new_entry], axis=0)
                text_head_weight[-1, :] = start_token_row
                gpt_checkpoint['text_head.weight'] = text_head_weight
                text_head_bias = gpt_checkpoint['text_head.bias']
                start_token_row = text_head_bias[-1]
                new_bias_entry = torch.zeros(num_new_tokens)
                text_head_bias = torch.cat([text_head_bias, new_bias_entry], axis=0)
                text_head_bias[-1] = start_token_row
                gpt_checkpoint['text_head.bias'] = text_head_bias
            self.xtts.gpt.load_state_dict(gpt_checkpoint, strict=True)
            print('>> GPT weights restored from:', self.args.gpt_checkpoint)
        if self.args.gpt_use_perceiver_resampler:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(filter_length=2048, hop_length=256, win_length=1024, normalize=False, sampling_rate=config.audio.sample_rate, mel_fmin=0, mel_fmax=8000, n_mel_channels=80, mel_norm_file=self.args.mel_norm_file)
        else:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(filter_length=4096, hop_length=1024, win_length=4096, normalize=False, sampling_rate=config.audio.sample_rate, mel_fmin=0, mel_fmax=8000, n_mel_channels=80, mel_norm_file=self.args.mel_norm_file)
        self.dvae = DiscreteVAE(channels=80, normalization=None, positional_dims=1, num_tokens=self.args.gpt_num_audio_tokens - 2, codebook_dim=512, hidden_dim=512, num_resnet_blocks=3, kernel_size=3, num_layers=2, use_transposed_convs=False)
        self.dvae.eval()
        if self.args.dvae_checkpoint:
            dvae_checkpoint = torch.load(self.args.dvae_checkpoint, map_location=torch.device('cpu'))
            self.dvae.load_state_dict(dvae_checkpoint, strict=False)
            print('>> DVAE weights restored from:', self.args.dvae_checkpoint)
        else:
            raise RuntimeError('You need to specify config.model_args.dvae_checkpoint path to be able to train the GPT decoder!!')
        self.torch_mel_spectrogram_dvae = TorchMelSpectrogram(mel_norm_file=self.args.mel_norm_file, sampling_rate=config.audio.dvae_sample_rate)

    @property
    def device(self):
        if False:
            i = 10
            return i + 15
        return next(self.parameters()).device

    def forward(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens):
        if False:
            while True:
                i = 10
        '\n        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode\n        (actuated by `text_first`).\n\n        text_inputs: long tensor, (b,t)\n        text_lengths: long tensor, (b,)\n        mel_inputs:  long tensor, (b,m)\n        wav_lengths: long tensor, (b,)\n        cond_mels: MEL float tensor, (b, num_samples, 80,t_m)\n        cond_idxs: cond start and end indexs, (b, 2)\n        cond_lens: long tensor, (b,)\n        '
        losses = self.xtts.gpt(text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels=cond_mels, cond_idxs=cond_idxs, cond_lens=cond_lens)
        return losses

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        if False:
            return 10
        if self.config.test_sentences:
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.xtts.gpt.eval()
            test_audios = {}
            print(' | > Synthesizing test sentences.')
            for (idx, s_info) in enumerate(self.config.test_sentences):
                wav = self.xtts.synthesize(s_info['text'], self.config, s_info['speaker_wav'], s_info['language'], gpt_cond_len=3)['wav']
                test_audios['{}-audio'.format(idx)] = wav
            del self.xtts.gpt.gpt_inference
            del self.xtts.gpt.gpt.wte
        return {'audios': test_audios}

    def test_log(self, outputs: dict, logger: 'Logger', assets: dict, steps: int) -> None:
        if False:
            i = 10
            return i + 15
        logger.test_audios(steps, outputs['audios'], self.args.output_sample_rate)

    def format_batch(self, batch: Dict) -> Dict:
        if False:
            while True:
                i = 10
        return batch

    @torch.no_grad()
    def format_batch_on_device(self, batch):
        if False:
            print('Hello World!')
        'Compute spectrograms on the device.'
        batch['text_lengths'] = batch['text_lengths']
        batch['wav_lengths'] = batch['wav_lengths']
        batch['text_inputs'] = batch['padded_text']
        batch['cond_idxs'] = batch['cond_idxs']
        (B, num_cond_samples, C, T) = batch['conditioning'].size()
        conditioning_reshaped = batch['conditioning'].view(B * num_cond_samples, C, T)
        paired_conditioning_mel = self.torch_mel_spectrogram_style_encoder(conditioning_reshaped)
        n_mel = self.torch_mel_spectrogram_style_encoder.n_mel_channels
        T_mel = paired_conditioning_mel.size(2)
        paired_conditioning_mel = paired_conditioning_mel.view(B, num_cond_samples, n_mel, T_mel)
        batch['cond_mels'] = paired_conditioning_mel
        if self.config.audio.sample_rate != self.config.audio.dvae_sample_rate:
            dvae_wav = torchaudio.functional.resample(batch['wav'], orig_freq=self.config.audio.sample_rate, new_freq=self.config.audio.dvae_sample_rate, lowpass_filter_width=64, rolloff=0.9475937167399596, resampling_method='kaiser_window', beta=14.769656459379492)
        else:
            dvae_wav = batch['wav']
        dvae_mel_spec = self.torch_mel_spectrogram_dvae(dvae_wav)
        codes = self.dvae.get_codebook_indices(dvae_mel_spec)
        batch['audio_codes'] = codes
        del batch['padded_text']
        del batch['wav']
        del batch['conditioning']
        return batch

    def train_step(self, batch, criterion):
        if False:
            print('Hello World!')
        loss_dict = {}
        cond_mels = batch['cond_mels']
        text_inputs = batch['text_inputs']
        text_lengths = batch['text_lengths']
        audio_codes = batch['audio_codes']
        wav_lengths = batch['wav_lengths']
        cond_idxs = batch['cond_idxs']
        cond_lens = batch['cond_lens']
        (loss_text, loss_mel, _) = self.forward(text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens)
        loss_dict['loss_text_ce'] = loss_text * self.args.gpt_loss_text_ce_weight
        loss_dict['loss_mel_ce'] = loss_mel * self.args.gpt_loss_mel_ce_weight
        loss_dict['loss'] = loss_dict['loss_text_ce'] + loss_dict['loss_mel_ce']
        return ({'model_outputs': None}, loss_dict)

    def eval_step(self, batch, criterion):
        if False:
            print('Hello World!')
        batch['cond_idxs'] = None
        return self.train_step(batch, criterion)

    def on_epoch_start(self, trainer):
        if False:
            for i in range(10):
                print('nop')
        self.dvae = self.dvae.eval()

    def on_init_end(self, trainer):
        if False:
            return 10
        if self.config.dashboard_logger.lower() == 'clearml':
            from clearml.binding.frameworks import WeightsFileHandler
            WeightsFileHandler.add_pre_callback(callback_clearml_load_save)

    @torch.no_grad()
    def inference(self, x, aux_input=None):
        if False:
            print('Hello World!')
        return None

    @staticmethod
    def get_criterion():
        if False:
            for i in range(10):
                print('nop')
        return None

    def get_sampler(self, dataset: TTSDataset, num_gpus=1):
        if False:
            print('Hello World!')
        batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        return batch_sampler

    def get_data_loader(self, config: Coqpit, assets: Dict, is_eval: bool, samples: Union[List[Dict], List[List]], verbose: bool, num_gpus: int, rank: int=None) -> 'DataLoader':
        if False:
            print('Hello World!')
        if is_eval and (not config.run_eval):
            loader = None
        else:
            dataset = XTTSDataset(self.config, samples, self.xtts.tokenizer, config.audio.sample_rate, is_eval)
            if num_gpus > 1:
                torch.distributed.barrier()
            sampler = self.get_sampler(dataset, num_gpus)
            if sampler is None or is_eval:
                loader = DataLoader(dataset, batch_size=config.eval_batch_size if is_eval else config.batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn, num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers, pin_memory=False)
            else:
                loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collate_fn, num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers, pin_memory=False)
        return loader

    def get_optimizer(self) -> List:
        if False:
            print('Hello World!')
        'Initiate and return the optimizer based on the config parameters.'
        if self.config.optimizer_wd_only_on_weights:
            net = self.xtts.gpt
            norm_modules = (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm1d, nn.InstanceNorm1d, nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm, nn.LayerNorm)
            emb_modules = (nn.Embedding, nn.EmbeddingBag)
            param_names_notweights = set()
            all_param_names = set()
            param_map = {}
            for (mn, m) in net.named_modules():
                for (k, v) in m.named_parameters():
                    v.is_bias = k.endswith('.bias')
                    v.is_weight = k.endswith('.weight')
                    v.is_norm = isinstance(m, norm_modules)
                    v.is_emb = isinstance(m, emb_modules)
                    fpn = '%s.%s' % (mn, k) if mn else k
                    all_param_names.add(fpn)
                    param_map[fpn] = v
                    if v.is_bias or v.is_norm or v.is_emb:
                        param_names_notweights.add(fpn)
            params_names_notweights = sorted(list(param_names_notweights))
            params_notweights = [param_map[k] for k in params_names_notweights]
            params_names_weights = sorted(list(all_param_names ^ param_names_notweights))
            params_weights = [param_map[k] for k in params_names_weights]
            groups = [{'params': params_weights, 'weight_decay': self.config.optimizer_params['weight_decay']}, {'params': params_notweights, 'weight_decay': 0}]
            opt = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr, parameters=groups)
            opt._group_names = [params_names_weights, params_names_notweights]
            return opt
        return get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr, parameters=self.xtts.gpt.parameters())

    def get_scheduler(self, optimizer) -> List:
        if False:
            for i in range(10):
                print('nop')
        'Set the scheduler for the optimizer.\n\n        Args:\n            optimizer: `torch.optim.Optimizer`.\n        '
        return get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)

    def load_checkpoint(self, config, checkpoint_path, eval=False, strict=True, cache_storage='/tmp/tts_cache', target_protocol='s3', target_options={'anon': True}):
        if False:
            i = 10
            return i + 15
        'Load the model checkpoint and setup for training or inference'
        state = self.xtts.get_compatible_checkpoint_state_dict(checkpoint_path)
        self.xtts.load_state_dict(state, strict=strict)
        if eval:
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.eval()
            assert not self.training

    @staticmethod
    def init_from_config(config: 'GPTTrainerConfig', samples: Union[List[List], List[Dict]]=None):
        if False:
            print('Hello World!')
        'Initiate model from config\n\n        Args:\n            config (GPTTrainerConfig): Model config.\n            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.\n                Defaults to None.\n        '
        return GPTTrainer(config)