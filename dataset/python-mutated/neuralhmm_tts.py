import os
from typing import Dict, List, Union
import torch
from coqpit import Coqpit
from torch import nn
from trainer.logging.tensorboard_logger import TensorboardLogger
from TTS.tts.layers.overflow.common_layers import Encoder, OverflowUtils
from TTS.tts.layers.overflow.neural_hmm import NeuralHMM
from TTS.tts.layers.overflow.plotting_utils import get_spec_from_most_probable_state, plot_transition_probabilities_to_numpy
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.generic_utils import format_aux_input
from TTS.utils.io import load_fsspec

class NeuralhmmTTS(BaseTTS):
    """Neural HMM TTS model.

    Paper::
        https://arxiv.org/abs/2108.13320

    Paper abstract::
        Neural sequence-to-sequence TTS has achieved significantly better output quality
    than statistical speech synthesis using HMMs.However, neural TTS is generally not probabilistic
    and uses non-monotonic attention. Attention failures increase training time and can make
    synthesis babble incoherently. This paper describes how the old and new paradigms can be
    combined to obtain the advantages of both worlds, by replacing attention in neural TTS with
    an autoregressive left-right no-skip hidden Markov model defined by a neural network.
    Based on this proposal, we modify Tacotron 2 to obtain an HMM-based neural TTS model with
    monotonic alignment, trained to maximise the full sequence likelihood without approximation.
    We also describe how to combine ideas from classical and contemporary TTS for best results.
    The resulting example system is smaller and simpler than Tacotron 2, and learns to speak with
    fewer iterations and less data, whilst achieving comparable naturalness prior to the post-net.
    Our approach also allows easy control over speaking rate. Audio examples and code
    are available at https://shivammehta25.github.io/Neural-HMM/ .

    Note:
        - This is a parameter efficient version of OverFlow (15.3M vs 28.6M). Since it has half the
        number of parameters as OverFlow the synthesis output quality is suboptimal (but comparable to Tacotron2
        without Postnet), but it learns to speak with even lesser amount of data and is still significantly faster
        than other attention-based methods.

        - Neural HMMs uses flat start initialization i.e it computes the means and std and transition probabilities
        of the dataset and uses them to initialize the model. This benefits the model and helps with faster learning
        If you change the dataset or want to regenerate the parameters change the `force_generate_statistics` and
        `mel_statistics_parameter_path` accordingly.

        - To enable multi-GPU training, set the `use_grad_checkpointing=False` in config.
        This will significantly increase the memory usage.  This is because to compute
        the actual data likelihood (not an approximation using MAS/Viterbi) we must use
        all the states at the previous time step during the forward pass to decide the
        probability distribution at the current step i.e the difference between the forward
        algorithm and viterbi approximation.

    Check :class:`TTS.tts.configs.neuralhmm_tts_config.NeuralhmmTTSConfig` for class arguments.
    """

    def __init__(self, config: 'NeuralhmmTTSConfig', ap: 'AudioProcessor'=None, tokenizer: 'TTSTokenizer'=None, speaker_manager: SpeakerManager=None):
        if False:
            print('Hello World!')
        super().__init__(config, ap, tokenizer, speaker_manager)
        self.config = config
        for key in config:
            setattr(self, key, config[key])
        self.encoder = Encoder(config.num_chars, config.state_per_phone, config.encoder_in_out_features)
        self.neural_hmm = NeuralHMM(frame_channels=self.out_channels, ar_order=self.ar_order, deterministic_transition=self.deterministic_transition, encoder_dim=self.encoder_in_out_features, prenet_type=self.prenet_type, prenet_dim=self.prenet_dim, prenet_n_layers=self.prenet_n_layers, prenet_dropout=self.prenet_dropout, prenet_dropout_at_inference=self.prenet_dropout_at_inference, memory_rnn_dim=self.memory_rnn_dim, outputnet_size=self.outputnet_size, flat_start_params=self.flat_start_params, std_floor=self.std_floor, use_grad_checkpointing=self.use_grad_checkpointing)
        self.register_buffer('mean', torch.tensor(0))
        self.register_buffer('std', torch.tensor(1))

    def update_mean_std(self, statistics_dict: Dict):
        if False:
            return 10
        self.mean.data = torch.tensor(statistics_dict['mean'])
        self.std.data = torch.tensor(statistics_dict['std'])

    def preprocess_batch(self, text, text_len, mels, mel_len):
        if False:
            i = 10
            return i + 15
        if self.mean.item() == 0 or self.std.item() == 1:
            statistics_dict = torch.load(self.mel_statistics_parameter_path)
            self.update_mean_std(statistics_dict)
        mels = self.normalize(mels)
        return (text, text_len, mels, mel_len)

    def normalize(self, x):
        if False:
            print('Hello World!')
        return x.sub(self.mean).div(self.std)

    def inverse_normalize(self, x):
        if False:
            while True:
                i = 10
        return x.mul(self.std).add(self.mean)

    def forward(self, text, text_len, mels, mel_len):
        if False:
            return 10
        '\n        Forward pass for training and computing the log likelihood of a given batch.\n\n        Shapes:\n            Shapes:\n            text: :math:`[B, T_in]`\n            text_len: :math:`[B]`\n            mels: :math:`[B, T_out, C]`\n            mel_len: :math:`[B]`\n        '
        (text, text_len, mels, mel_len) = self.preprocess_batch(text, text_len, mels, mel_len)
        (encoder_outputs, encoder_output_len) = self.encoder(text, text_len)
        (log_probs, fwd_alignments, transition_vectors, means) = self.neural_hmm(encoder_outputs, encoder_output_len, mels.transpose(1, 2), mel_len)
        outputs = {'log_probs': log_probs, 'alignments': fwd_alignments, 'transition_vectors': transition_vectors, 'means': means}
        return outputs

    @staticmethod
    def _training_stats(batch):
        if False:
            while True:
                i = 10
        stats = {}
        stats['avg_text_length'] = batch['text_lengths'].float().mean()
        stats['avg_spec_length'] = batch['mel_lengths'].float().mean()
        stats['avg_text_batch_occupancy'] = (batch['text_lengths'].float() / batch['text_lengths'].float().max()).mean()
        stats['avg_spec_batch_occupancy'] = (batch['mel_lengths'].float() / batch['mel_lengths'].float().max()).mean()
        return stats

    def train_step(self, batch: dict, criterion: nn.Module):
        if False:
            while True:
                i = 10
        text_input = batch['text_input']
        text_lengths = batch['text_lengths']
        mel_input = batch['mel_input']
        mel_lengths = batch['mel_lengths']
        outputs = self.forward(text=text_input, text_len=text_lengths, mels=mel_input, mel_len=mel_lengths)
        loss_dict = criterion(outputs['log_probs'] / (mel_lengths.sum() + text_lengths.sum()))
        loss_dict.update(self._training_stats(batch))
        return (outputs, loss_dict)

    def eval_step(self, batch: Dict, criterion: nn.Module):
        if False:
            return 10
        return self.train_step(batch, criterion)

    def _format_aux_input(self, aux_input: Dict, default_input_dict):
        if False:
            for i in range(10):
                print('nop')
        'Set missing fields to their default value.\n\n        Args:\n            aux_inputs (Dict): Dictionary containing the auxiliary inputs.\n        '
        default_input_dict = default_input_dict.copy()
        default_input_dict.update({'sampling_temp': self.sampling_temp, 'max_sampling_time': self.max_sampling_time, 'duration_threshold': self.duration_threshold})
        if aux_input:
            return format_aux_input(default_input_dict, aux_input)
        return default_input_dict

    @torch.no_grad()
    def inference(self, text: torch.Tensor, aux_input={'x_lengths': None, 'sampling_temp': None, 'max_sampling_time': None, 'duration_threshold': None}):
        if False:
            return 10
        'Sampling from the model\n\n        Args:\n            text (torch.Tensor): :math:`[B, T_in]`\n            aux_inputs (_type_, optional): _description_. Defaults to None.\n\n        Returns:\n            outputs: Dictionary containing the following\n                - mel (torch.Tensor): :math:`[B, T_out, C]`\n                - hmm_outputs_len (torch.Tensor): :math:`[B]`\n                - state_travelled (List[List[int]]): List of lists containing the state travelled for each sample in the batch.\n                - input_parameters (list[torch.FloatTensor]): Input parameters to the neural HMM.\n                - output_parameters (list[torch.FloatTensor]): Output parameters to the neural HMM.\n        '
        default_input_dict = {'x_lengths': torch.sum(text != 0, dim=1)}
        aux_input = self._format_aux_input(aux_input, default_input_dict)
        (encoder_outputs, encoder_output_len) = self.encoder.inference(text, aux_input['x_lengths'])
        outputs = self.neural_hmm.inference(encoder_outputs, encoder_output_len, sampling_temp=aux_input['sampling_temp'], max_sampling_time=aux_input['max_sampling_time'], duration_threshold=aux_input['duration_threshold'])
        (mels, mel_outputs_len) = (outputs['hmm_outputs'], outputs['hmm_outputs_len'])
        mels = self.inverse_normalize(mels)
        outputs.update({'model_outputs': mels, 'model_outputs_len': mel_outputs_len})
        outputs['alignments'] = OverflowUtils.double_pad(outputs['alignments'])
        return outputs

    @staticmethod
    def get_criterion():
        if False:
            print('Hello World!')
        return NLLLoss()

    @staticmethod
    def init_from_config(config: 'NeuralhmmTTSConfig', samples: Union[List[List], List[Dict]]=None, verbose=True):
        if False:
            print('Hello World!')
        'Initiate model from config\n\n        Args:\n            config (VitsConfig): Model config.\n            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.\n                Defaults to None.\n            verbose (bool): If True, print init messages. Defaults to True.\n        '
        from TTS.utils.audio import AudioProcessor
        ap = AudioProcessor.init_from_config(config, verbose)
        (tokenizer, new_config) = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        return NeuralhmmTTS(new_config, ap, tokenizer, speaker_manager)

    def load_checkpoint(self, config: Coqpit, checkpoint_path: str, eval: bool=False, strict: bool=True, cache=False):
        if False:
            while True:
                i = 10
        state = load_fsspec(checkpoint_path, map_location=torch.device('cpu'))
        self.load_state_dict(state['model'])
        if eval:
            self.eval()
            assert not self.training

    def on_init_start(self, trainer):
        if False:
            print('Hello World!')
        'If the current dataset does not have normalisation statistics and initialisation transition_probability it computes them otherwise loads.'
        if not os.path.isfile(trainer.config.mel_statistics_parameter_path) or trainer.config.force_generate_statistics:
            dataloader = trainer.get_train_dataloader(training_assets=None, samples=trainer.train_samples, verbose=False)
            print(f' | > Data parameters not found for: {trainer.config.mel_statistics_parameter_path}. Computing mel normalization parameters...')
            (data_mean, data_std, init_transition_prob) = OverflowUtils.get_data_parameters_for_flat_start(dataloader, trainer.config.out_channels, trainer.config.state_per_phone)
            print(f' | > Saving data parameters to: {trainer.config.mel_statistics_parameter_path}: value: {(data_mean, data_std, init_transition_prob)}')
            statistics = {'mean': data_mean.item(), 'std': data_std.item(), 'init_transition_prob': init_transition_prob.item()}
            torch.save(statistics, trainer.config.mel_statistics_parameter_path)
        else:
            print(f' | > Data parameters found for: {trainer.config.mel_statistics_parameter_path}. Loading mel normalization parameters...')
            statistics = torch.load(trainer.config.mel_statistics_parameter_path)
            (data_mean, data_std, init_transition_prob) = (statistics['mean'], statistics['std'], statistics['init_transition_prob'])
            print(f' | > Data parameters loaded with value: {(data_mean, data_std, init_transition_prob)}')
        trainer.config.flat_start_params['transition_p'] = init_transition_prob.item() if torch.is_tensor(init_transition_prob) else init_transition_prob
        OverflowUtils.update_flat_start_transition(trainer.model, init_transition_prob)
        trainer.model.update_mean_std(statistics)

    @torch.inference_mode()
    def _create_logs(self, batch, outputs, ap):
        if False:
            while True:
                i = 10
        (alignments, transition_vectors) = (outputs['alignments'], outputs['transition_vectors'])
        means = torch.stack(outputs['means'], dim=1)
        figures = {'alignment': plot_alignment(alignments[0].exp(), title='Forward alignment', fig_size=(20, 20)), 'log_alignment': plot_alignment(alignments[0].exp(), title='Forward log alignment', plot_log=True, fig_size=(20, 20)), 'transition_vectors': plot_alignment(transition_vectors[0], title='Transition vectors', fig_size=(20, 20)), 'mel_from_most_probable_state': plot_spectrogram(get_spec_from_most_probable_state(alignments[0], means[0]), fig_size=(12, 3)), 'mel_target': plot_spectrogram(batch['mel_input'][0], fig_size=(12, 3))}
        print(' | > Synthesising audio from the model...')
        inference_output = self.inference(batch['text_input'][-1].unsqueeze(0), aux_input={'x_lengths': batch['text_lengths'][-1].unsqueeze(0)})
        figures['synthesised'] = plot_spectrogram(inference_output['model_outputs'][0], fig_size=(12, 3))
        states = [p[1] for p in inference_output['input_parameters'][0]]
        transition_probability_synthesising = [p[2].cpu().numpy() for p in inference_output['output_parameters'][0]]
        for i in range(len(transition_probability_synthesising) // 200 + 1):
            start = i * 200
            end = (i + 1) * 200
            figures[f'synthesised_transition_probabilities/{i}'] = plot_transition_probabilities_to_numpy(states[start:end], transition_probability_synthesising[start:end])
        audio = ap.inv_melspectrogram(inference_output['model_outputs'][0].T.cpu().numpy())
        return (figures, {'audios': audio})

    def train_log(self, batch: dict, outputs: dict, logger: 'Logger', assets: dict, steps: int):
        if False:
            for i in range(10):
                print('nop')
        'Log training progress.'
        (figures, audios) = self._create_logs(batch, outputs, self.ap)
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    def eval_log(self, batch: Dict, outputs: Dict, logger: 'Logger', assets: Dict, steps: int):
        if False:
            while True:
                i = 10
        'Compute and log evaluation metrics.'
        if isinstance(logger, TensorboardLogger):
            for (tag, value) in self.named_parameters():
                tag = tag.replace('.', '/')
                logger.writer.add_histogram(tag, value.data.cpu().numpy(), steps)
        (figures, audios) = self._create_logs(batch, outputs, self.ap)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    def test_log(self, outputs: dict, logger: 'Logger', assets: dict, steps: int) -> None:
        if False:
            print('Hello World!')
        logger.test_audios(steps, outputs[1], self.ap.sample_rate)
        logger.test_figures(steps, outputs[0])

class NLLLoss(nn.Module):
    """Negative log likelihood loss."""

    def forward(self, log_prob: torch.Tensor) -> dict:
        if False:
            for i in range(10):
                print('nop')
        'Compute the loss.\n\n        Args:\n            logits (Tensor): [B, T, D]\n\n        Returns:\n            Tensor: [1]\n\n        '
        return_dict = {}
        return_dict['loss'] = -log_prob.mean()
        return return_dict