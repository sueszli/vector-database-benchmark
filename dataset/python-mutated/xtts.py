import os
from dataclasses import dataclass
import librosa
import torch
import torch.nn.functional as F
import torchaudio
from coqpit import Coqpit
from TTS.tts.layers.xtts.gpt import GPT
from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder
from TTS.tts.layers.xtts.stream_generator import init_stream_support
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.models.base_tts import BaseTTS
from TTS.utils.io import load_fsspec
init_stream_support()

def wav_to_mel_cloning(wav, mel_norms_file='../experiments/clips_mel_norms.pth', mel_norms=None, device=torch.device('cpu'), n_fft=4096, hop_length=1024, win_length=4096, power=2, normalized=False, sample_rate=22050, f_min=0, f_max=8000, n_mels=80):
    if False:
        return 10
    '\n    Convert waveform to mel-spectrogram with hard-coded parameters for cloning.\n\n    Args:\n        wav (torch.Tensor): Input waveform tensor.\n        mel_norms_file (str): Path to mel-spectrogram normalization file.\n        mel_norms (torch.Tensor): Mel-spectrogram normalization tensor.\n        device (torch.device): Device to use for computation.\n\n    Returns:\n        torch.Tensor: Mel-spectrogram tensor.\n    '
    mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=power, normalized=normalized, sample_rate=sample_rate, f_min=f_min, f_max=f_max, n_mels=n_mels, norm='slaney').to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-05))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device)
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel

def load_audio(audiopath, sampling_rate):
    if False:
        for i in range(10):
            print('nop')
    (audio, lsr) = torchaudio.load(audiopath)
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    if torch.any(audio > 10) or not torch.any(audio < 0):
        print(f'Error with {audiopath}. Max={audio.max()} min={audio.min()}')
    audio.clip_(-1, 1)
    return audio

def pad_or_truncate(t, length):
    if False:
        return 10
    '\n    Ensure a given tensor t has a specified sequence length by either padding it with zeros or clipping it.\n\n    Args:\n        t (torch.Tensor): The input tensor to be padded or truncated.\n        length (int): The desired length of the tensor.\n\n    Returns:\n        torch.Tensor: The padded or truncated tensor.\n    '
    tp = t[..., :length]
    if t.shape[-1] == length:
        tp = t
    elif t.shape[-1] < length:
        tp = F.pad(t, (0, length - t.shape[-1]))
    return tp

@dataclass
class XttsAudioConfig(Coqpit):
    """
    Configuration class for audio-related parameters in the XTTS model.

    Args:
        sample_rate (int): The sample rate in which the GPT operates.
        output_sample_rate (int): The sample rate of the output audio waveform.
    """
    sample_rate: int = 22050
    output_sample_rate: int = 24000

@dataclass
class XttsArgs(Coqpit):
    """A dataclass to represent XTTS model arguments that define the model structure.

    Args:
        gpt_batch_size (int): The size of the auto-regressive batch.
        enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
        clvp_checkpoint (str, optional): The checkpoint for the ConditionalLatentVariablePerseq model. Defaults to None.
        decoder_checkpoint (str, optional): The checkpoint for the DiffTTS model. Defaults to None.
        num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

        For GPT model:
        gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
        gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
        gpt_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.
        gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
        gpt_use_masking_gt_prompt_approach (bool, optional):  If True, it will use ground truth as prompt and it will mask the loss to avoid repetition. Defaults to True.
        gpt_use_perceiver_resampler (bool, optional):  If True, it will use perceiver resampler from flamingo paper - https://arxiv.org/abs/2204.14198. Defaults to False.
    """
    gpt_batch_size: int = 1
    enable_redaction: bool = False
    kv_cache: bool = True
    gpt_checkpoint: str = None
    clvp_checkpoint: str = None
    decoder_checkpoint: str = None
    num_chars: int = 255
    tokenizer_file: str = ''
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = None
    gpt_start_text_token: int = None
    gpt_stop_text_token: int = None
    gpt_num_audio_tokens: int = 8194
    gpt_start_audio_token: int = 8192
    gpt_stop_audio_token: int = 8193
    gpt_code_stride_len: int = 1024
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = False
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True
    duration_const: int = 102400

class Xtts(BaseTTS):
    """ⓍTTS model implementation.

    ❗ Currently it only supports inference.

    Examples:
        >>> from TTS.tts.configs.xtts_config import XttsConfig
        >>> from TTS.tts.models.xtts import Xtts
        >>> config = XttsConfig()
        >>> model = Xtts.inif_from_config(config)
        >>> model.load_checkpoint(config, checkpoint_dir="paths/to/models_dir/", eval=True)
    """

    def __init__(self, config: Coqpit):
        if False:
            i = 10
            return i + 15
        super().__init__(config, ap=None, tokenizer=None)
        self.mel_stats_path = None
        self.config = config
        self.gpt_checkpoint = self.args.gpt_checkpoint
        self.decoder_checkpoint = self.args.decoder_checkpoint
        self.models_dir = config.model_dir
        self.gpt_batch_size = self.args.gpt_batch_size
        self.tokenizer = VoiceBpeTokenizer()
        self.gpt = None
        self.init_models()
        self.register_buffer('mel_stats', torch.ones(80))

    def init_models(self):
        if False:
            print('Hello World!')
        'Initialize the models. We do it here since we need to load the tokenizer first.'
        if self.tokenizer.tokenizer is not None:
            self.args.gpt_number_text_tokens = self.tokenizer.get_number_tokens()
            self.args.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id('[START]')
            self.args.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id('[STOP]')
        if self.args.gpt_number_text_tokens:
            self.gpt = GPT(layers=self.args.gpt_layers, model_dim=self.args.gpt_n_model_channels, start_text_token=self.args.gpt_start_text_token, stop_text_token=self.args.gpt_stop_text_token, heads=self.args.gpt_n_heads, max_text_tokens=self.args.gpt_max_text_tokens, max_mel_tokens=self.args.gpt_max_audio_tokens, max_prompt_tokens=self.args.gpt_max_prompt_tokens, number_text_tokens=self.args.gpt_number_text_tokens, num_audio_tokens=self.args.gpt_num_audio_tokens, start_audio_token=self.args.gpt_start_audio_token, stop_audio_token=self.args.gpt_stop_audio_token, use_perceiver_resampler=self.args.gpt_use_perceiver_resampler, code_stride_len=self.args.gpt_code_stride_len)
        self.hifigan_decoder = HifiDecoder(input_sample_rate=self.args.input_sample_rate, output_sample_rate=self.args.output_sample_rate, output_hop_length=self.args.output_hop_length, ar_mel_length_compression=self.args.gpt_code_stride_len, decoder_input_dim=self.args.decoder_input_dim, d_vector_dim=self.args.d_vector_dim, cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer)

    @property
    def device(self):
        if False:
            for i in range(10):
                print('nop')
        return next(self.parameters()).device

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int=30, chunk_length: int=6):
        if False:
            i = 10
            return i + 15
        'Compute the conditioning latents for the GPT model from the given audio.\n\n        Args:\n            audio (tensor): audio tensor.\n            sr (int): Sample rate of the audio.\n            length (int): Length of the audio in seconds. If < 0, use the whole audio. Defaults to 30.\n            chunk_length (int): Length of the audio chunks in seconds. When `length == chunk_length`, the whole audio\n                is being used without chunking. It must be < `length`. Defaults to 6.\n        '
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, :22050 * length]
        if self.args.gpt_use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i:i + 22050 * chunk_length]
                mel_chunk = wav_to_mel_cloning(audio_chunk, mel_norms=self.mel_stats.cpu(), n_fft=2048, hop_length=256, win_length=1024, power=2, normalized=False, sample_rate=22050, f_min=0, f_max=8000, n_mels=80)
                style_emb = self.gpt.get_style_emb(mel_chunk.to(self.device), None)
                style_embs.append(style_emb)
            cond_latent = torch.stack(style_embs).mean(dim=0)
        else:
            mel = wav_to_mel_cloning(audio, mel_norms=self.mel_stats.cpu(), n_fft=4096, hop_length=1024, win_length=4096, power=2, normalized=False, sample_rate=22050, f_min=0, f_max=8000, n_mels=80)
            cond_latent = self.gpt.get_style_emb(mel.to(self.device))
        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        if False:
            print('Hello World!')
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True).unsqueeze(-1).to(self.device)

    @torch.inference_mode()
    def get_conditioning_latents(self, audio_path, max_ref_length=30, gpt_cond_len=6, gpt_cond_chunk_len=6, librosa_trim_db=None, sound_norm_refs=False, load_sr=22050):
        if False:
            print('Hello World!')
        'Get the conditioning latents for the GPT model from the given audio.\n\n        Args:\n            audio_path (str or List[str]): Path to reference audio file(s).\n            max_ref_length (int): Maximum length of each reference audio in seconds. Defaults to 30.\n            gpt_cond_len (int): Length of the audio used for gpt latents. Defaults to 6.\n            gpt_cond_chunk_len (int): Chunk length used for gpt latents. It must be <= gpt_conf_len. Defaults to 6.\n            librosa_trim_db (int, optional): Trim the audio using this value. If None, not trimming. Defaults to None.\n            sound_norm_refs (bool, optional): Whether to normalize the audio. Defaults to False.\n            load_sr (int, optional): Sample rate to load the audio. Defaults to 24000.\n        '
        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path
        speaker_embeddings = []
        audios = []
        speaker_embedding = None
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr)
            audio = audio[:, :load_sr * max_ref_length].to(self.device)
            if sound_norm_refs:
                audio = audio / torch.abs(audio).max() * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]
            speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(speaker_embedding)
            audios.append(audio)
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len)
        if speaker_embeddings:
            speaker_embedding = torch.stack(speaker_embeddings)
            speaker_embedding = speaker_embedding.mean(dim=0)
        return (gpt_cond_latents, speaker_embedding)

    def synthesize(self, text, config, speaker_wav, language, **kwargs):
        if False:
            while True:
                i = 10
        'Synthesize speech with the given input text.\n\n        Args:\n            text (str): Input text.\n            config (XttsConfig): Config with inference parameters.\n            speaker_wav (list): List of paths to the speaker audio files to be used for cloning.\n            language (str): Language ID of the speaker.\n            **kwargs: Inference settings. See `inference()`.\n\n        Returns:\n            A dictionary of the output values with `wav` as output waveform, `deterministic_seed` as seed used at inference,\n            `text_input` as text token IDs after tokenizer, `voice_samples` as samples used for cloning, `conditioning_latents`\n            as latents used at inference.\n\n        '
        return self.inference_with_config(text, config, ref_audio_path=speaker_wav, language=language, **kwargs)

    def inference_with_config(self, text, config, ref_audio_path, language, **kwargs):
        if False:
            return 10
        '\n        inference with config\n        '
        assert language in self.config.languages, f' ❗ Language {language} is not supported. Supported languages are {self.config.languages}'
        settings = {'temperature': config.temperature, 'length_penalty': config.length_penalty, 'repetition_penalty': config.repetition_penalty, 'top_k': config.top_k, 'top_p': config.top_p, 'gpt_cond_len': config.gpt_cond_len, 'gpt_cond_chunk_len': config.gpt_cond_chunk_len, 'max_ref_len': config.max_ref_len, 'sound_norm_refs': config.sound_norm_refs}
        settings.update(kwargs)
        return self.full_inference(text, ref_audio_path, language, **settings)

    @torch.inference_mode()
    def full_inference(self, text, ref_audio_path, language, temperature=0.65, length_penalty=1, repetition_penalty=2.0, top_k=50, top_p=0.85, do_sample=True, gpt_cond_len=30, gpt_cond_chunk_len=6, max_ref_len=10, sound_norm_refs=False, **hf_generate_kwargs):
        if False:
            return 10
        '\n        This function produces an audio clip of the given text being spoken with the given reference voice.\n\n        Args:\n            text: (str) Text to be spoken.\n\n            ref_audio_path: (str) Path to a reference audio file to be used for cloning. This audio file should be >3\n                seconds long.\n\n            language: (str) Language of the voice to be generated.\n\n            temperature: (float) The softmax temperature of the autoregressive model. Defaults to 0.65.\n\n            length_penalty: (float) A length penalty applied to the autoregressive decoder. Higher settings causes the\n                model to produce more terse outputs. Defaults to 1.0.\n\n            repetition_penalty: (float) A penalty that prevents the autoregressive decoder from repeating itself during\n                decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc. Defaults to 2.0.\n\n            top_k: (int) K value used in top-k sampling. [0,inf]. Lower values mean the decoder produces more "likely"\n                (aka boring) outputs. Defaults to 50.\n\n            top_p: (float) P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely"\n                (aka boring) outputs. Defaults to 0.8.\n\n            gpt_cond_len: (int) Length of the audio used for cloning. If audio is shorter, then audio length is used\n                else the first `gpt_cond_len` secs is used. Defaults to 30 seconds.\n\n            gpt_cond_chunk_len: (int) Chunk length used for cloning. It must be <= `gpt_cond_len`.\n                If gpt_cond_len == gpt_cond_chunk_len, no chunking. Defaults to 6 seconds.\n\n            hf_generate_kwargs: (**kwargs) The huggingface Transformers generate API is used for the autoregressive\n                transformer. Extra keyword args fed to this function get forwarded directly to that API. Documentation\n                here: https://huggingface.co/docs/transformers/internal/generation_utils\n\n        Returns:\n            Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.\n            Sample rate is 24kHz.\n        '
        (gpt_cond_latent, speaker_embedding) = self.get_conditioning_latents(audio_path=ref_audio_path, gpt_cond_len=gpt_cond_len, gpt_cond_chunk_len=gpt_cond_chunk_len, max_ref_length=max_ref_len, sound_norm_refs=sound_norm_refs)
        return self.inference(text, language, gpt_cond_latent, speaker_embedding, temperature=temperature, length_penalty=length_penalty, repetition_penalty=repetition_penalty, top_k=top_k, top_p=top_p, do_sample=do_sample, **hf_generate_kwargs)

    @torch.inference_mode()
    def inference(self, text, language, gpt_cond_latent, speaker_embedding, temperature=0.65, length_penalty=1, repetition_penalty=2.0, top_k=50, top_p=0.85, do_sample=True, num_beams=1, speed=1.0, **hf_generate_kwargs):
        if False:
            i = 10
            return i + 15
        length_scale = 1.0 / max(speed, 0.05)
        text = text.strip().lower()
        text_tokens = torch.IntTensor(self.tokenizer.encode(text, lang=language)).unsqueeze(0).to(self.device)
        assert text_tokens.shape[-1] < self.args.gpt_max_text_tokens, ' ❗ XTTS can only generate text with a maximum of 400 tokens.'
        with torch.no_grad():
            gpt_codes = self.gpt.generate(cond_latents=gpt_cond_latent, text_inputs=text_tokens, input_tokens=None, do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature, num_return_sequences=self.gpt_batch_size, num_beams=num_beams, length_penalty=length_penalty, repetition_penalty=repetition_penalty, output_attentions=False, **hf_generate_kwargs)
            expected_output_len = torch.tensor([gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device)
            text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
            gpt_latents = self.gpt(text_tokens, text_len, gpt_codes, expected_output_len, cond_latents=gpt_cond_latent, return_attentions=False, return_latent=True)
            if length_scale != 1.0:
                gpt_latents = F.interpolate(gpt_latents.transpose(1, 2), scale_factor=length_scale, mode='linear').transpose(1, 2)
            wav = self.hifigan_decoder(gpt_latents, g=speaker_embedding)
        return {'wav': wav.cpu().numpy().squeeze(), 'gpt_latents': gpt_latents, 'speaker_embedding': speaker_embedding}

    def handle_chunks(self, wav_gen, wav_gen_prev, wav_overlap, overlap_len):
        if False:
            print('Hello World!')
        'Handle chunk formatting in streaming mode'
        wav_chunk = wav_gen[:-overlap_len]
        if wav_gen_prev is not None:
            wav_chunk = wav_gen[wav_gen_prev.shape[0] - overlap_len:-overlap_len]
        if wav_overlap is not None:
            if overlap_len > len(wav_chunk):
                if wav_gen_prev is not None:
                    wav_chunk = wav_gen[wav_gen_prev.shape[0] - overlap_len:]
                else:
                    wav_chunk = wav_gen[-overlap_len:]
                return (wav_chunk, wav_gen, None)
            else:
                crossfade_wav = wav_chunk[:overlap_len]
                crossfade_wav = crossfade_wav * torch.linspace(0.0, 1.0, overlap_len).to(crossfade_wav.device)
                wav_chunk[:overlap_len] = wav_overlap * torch.linspace(1.0, 0.0, overlap_len).to(wav_overlap.device)
                wav_chunk[:overlap_len] += crossfade_wav
        wav_overlap = wav_gen[-overlap_len:]
        wav_gen_prev = wav_gen
        return (wav_chunk, wav_gen_prev, wav_overlap)

    @torch.inference_mode()
    def inference_stream(self, text, language, gpt_cond_latent, speaker_embedding, stream_chunk_size=20, overlap_wav_len=1024, temperature=0.65, length_penalty=1, repetition_penalty=2.0, top_k=50, top_p=0.85, do_sample=True, speed=1.0, **hf_generate_kwargs):
        if False:
            for i in range(10):
                print('nop')
        length_scale = 1.0 / max(speed, 0.05)
        text = text.strip().lower()
        text_tokens = torch.IntTensor(self.tokenizer.encode(text, lang=language)).unsqueeze(0).to(self.device)
        fake_inputs = self.gpt.compute_embeddings(gpt_cond_latent.to(self.device), text_tokens)
        gpt_generator = self.gpt.get_generator(fake_inputs=fake_inputs, top_k=top_k, top_p=top_p, temperature=temperature, do_sample=do_sample, num_beams=1, num_return_sequences=1, length_penalty=float(length_penalty), repetition_penalty=float(repetition_penalty), output_attentions=False, output_hidden_states=True, **hf_generate_kwargs)
        last_tokens = []
        all_latents = []
        wav_gen_prev = None
        wav_overlap = None
        is_end = False
        while not is_end:
            try:
                (x, latent) = next(gpt_generator)
                last_tokens += [x]
                all_latents += [latent]
            except StopIteration:
                is_end = True
            if is_end or (stream_chunk_size > 0 and len(last_tokens) >= stream_chunk_size):
                gpt_latents = torch.cat(all_latents, dim=0)[None, :]
                if length_scale != 1.0:
                    gpt_latents = F.interpolate(gpt_latents.transpose(1, 2), scale_factor=length_scale, mode='linear').transpose(1, 2)
                wav_gen = self.hifigan_decoder(gpt_latents, g=speaker_embedding.to(self.device))
                (wav_chunk, wav_gen_prev, wav_overlap) = self.handle_chunks(wav_gen.squeeze(), wav_gen_prev, wav_overlap, overlap_wav_len)
                last_tokens = []
                yield wav_chunk

    def forward(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training')

    def eval_step(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training')

    @staticmethod
    def init_from_config(config: 'XttsConfig', **kwargs):
        if False:
            i = 10
            return i + 15
        return Xtts(config)

    def eval(self):
        if False:
            print('Hello World!')
        'Sets the model to evaluation mode. Overrides the default eval() method to also set the GPT model to eval mode.'
        self.gpt.init_gpt_for_inference()
        super().eval()

    def get_compatible_checkpoint_state_dict(self, model_path):
        if False:
            while True:
                i = 10
        checkpoint = load_fsspec(model_path, map_location=torch.device('cpu'))['model']
        ignore_keys = ['torch_mel_spectrogram_style_encoder', 'torch_mel_spectrogram_dvae', 'dvae']
        for key in list(checkpoint.keys()):
            if key.startswith('xtts.'):
                new_key = key.replace('xtts.', '')
                checkpoint[new_key] = checkpoint[key]
                del checkpoint[key]
                key = new_key
            if key.split('.')[0] in ignore_keys:
                del checkpoint[key]
        return checkpoint

    def load_checkpoint(self, config, checkpoint_dir=None, checkpoint_path=None, vocab_path=None, eval=True, strict=True, use_deepspeed=False):
        if False:
            while True:
                i = 10
        "\n        Loads a checkpoint from disk and initializes the model's state and tokenizer.\n\n        Args:\n            config (dict): The configuration dictionary for the model.\n            checkpoint_dir (str, optional): The directory where the checkpoint is stored. Defaults to None.\n            checkpoint_path (str, optional): The path to the checkpoint file. Defaults to None.\n            vocab_path (str, optional): The path to the vocabulary file. Defaults to None.\n            eval (bool, optional): Whether to set the model to evaluation mode. Defaults to True.\n            strict (bool, optional): Whether to strictly enforce that the keys in the checkpoint match the keys in the model. Defaults to True.\n\n        Returns:\n            None\n        "
        model_path = checkpoint_path or os.path.join(checkpoint_dir, 'model.pth')
        vocab_path = vocab_path or os.path.join(checkpoint_dir, 'vocab.json')
        if os.path.exists(vocab_path):
            self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
        self.init_models()
        checkpoint = self.get_compatible_checkpoint_state_dict(model_path)
        try:
            self.load_state_dict(checkpoint, strict=strict)
        except:
            if eval:
                self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache)
            self.load_state_dict(checkpoint, strict=strict)
        if eval:
            self.hifigan_decoder.eval()
            self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=use_deepspeed)
            self.gpt.eval()

    def train_step(self):
        if False:
            return 10
        raise NotImplementedError('XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training')