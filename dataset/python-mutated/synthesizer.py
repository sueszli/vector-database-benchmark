import os
import time
from typing import List
import numpy as np
import pysbd
import torch
from torch import nn
from TTS.config import load_config
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models import setup_model as setup_tts_model
from TTS.tts.models.vits import Vits
from TTS.tts.utils.synthesis import synthesis, transfer_voice, trim_silence
from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.vc.models import setup_model as setup_vc_model
from TTS.vocoder.models import setup_model as setup_vocoder_model
from TTS.vocoder.utils.generic_utils import interpolate_vocoder_input

class Synthesizer(nn.Module):

    def __init__(self, tts_checkpoint: str='', tts_config_path: str='', tts_speakers_file: str='', tts_languages_file: str='', vocoder_checkpoint: str='', vocoder_config: str='', encoder_checkpoint: str='', encoder_config: str='', vc_checkpoint: str='', vc_config: str='', model_dir: str='', voice_dir: str=None, use_cuda: bool=False) -> None:
        if False:
            while True:
                i = 10
        'General 🐸 TTS interface for inference. It takes a tts and a vocoder\n        model and synthesize speech from the provided text.\n\n        The text is divided into a list of sentences using `pysbd` and synthesize\n        speech on each sentence separately.\n\n        If you have certain special characters in your text, you need to handle\n        them before providing the text to Synthesizer.\n\n        TODO: set the segmenter based on the source language\n\n        Args:\n            tts_checkpoint (str, optional): path to the tts model file.\n            tts_config_path (str, optional): path to the tts config file.\n            vocoder_checkpoint (str, optional): path to the vocoder model file. Defaults to None.\n            vocoder_config (str, optional): path to the vocoder config file. Defaults to None.\n            encoder_checkpoint (str, optional): path to the speaker encoder model file. Defaults to `""`,\n            encoder_config (str, optional): path to the speaker encoder config file. Defaults to `""`,\n            vc_checkpoint (str, optional): path to the voice conversion model file. Defaults to `""`,\n            vc_config (str, optional): path to the voice conversion config file. Defaults to `""`,\n            use_cuda (bool, optional): enable/disable cuda. Defaults to False.\n        '
        super().__init__()
        self.tts_checkpoint = tts_checkpoint
        self.tts_config_path = tts_config_path
        self.tts_speakers_file = tts_speakers_file
        self.tts_languages_file = tts_languages_file
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_config = vocoder_config
        self.encoder_checkpoint = encoder_checkpoint
        self.encoder_config = encoder_config
        self.vc_checkpoint = vc_checkpoint
        self.vc_config = vc_config
        self.use_cuda = use_cuda
        self.tts_model = None
        self.vocoder_model = None
        self.vc_model = None
        self.speaker_manager = None
        self.tts_speakers = {}
        self.language_manager = None
        self.num_languages = 0
        self.tts_languages = {}
        self.d_vector_dim = 0
        self.seg = self._get_segmenter('en')
        self.use_cuda = use_cuda
        self.voice_dir = voice_dir
        if self.use_cuda:
            assert torch.cuda.is_available(), 'CUDA is not availabe on this machine.'
        if tts_checkpoint:
            self._load_tts(tts_checkpoint, tts_config_path, use_cuda)
            self.output_sample_rate = self.tts_config.audio['sample_rate']
        if vocoder_checkpoint:
            self._load_vocoder(vocoder_checkpoint, vocoder_config, use_cuda)
            self.output_sample_rate = self.vocoder_config.audio['sample_rate']
        if vc_checkpoint:
            self._load_vc(vc_checkpoint, vc_config, use_cuda)
            self.output_sample_rate = self.vc_config.audio['output_sample_rate']
        if model_dir:
            if 'fairseq' in model_dir:
                self._load_fairseq_from_dir(model_dir, use_cuda)
                self.output_sample_rate = self.tts_config.audio['sample_rate']
            else:
                self._load_tts_from_dir(model_dir, use_cuda)
                self.output_sample_rate = self.tts_config.audio['output_sample_rate']

    @staticmethod
    def _get_segmenter(lang: str):
        if False:
            for i in range(10):
                print('nop')
        'get the sentence segmenter for the given language.\n\n        Args:\n            lang (str): target language code.\n\n        Returns:\n            [type]: [description]\n        '
        return pysbd.Segmenter(language=lang, clean=True)

    def _load_vc(self, vc_checkpoint: str, vc_config_path: str, use_cuda: bool) -> None:
        if False:
            return 10
        'Load the voice conversion model.\n\n        1. Load the model config.\n        2. Init the model from the config.\n        3. Load the model weights.\n        4. Move the model to the GPU if CUDA is enabled.\n\n        Args:\n            vc_checkpoint (str): path to the model checkpoint.\n            tts_config_path (str): path to the model config file.\n            use_cuda (bool): enable/disable CUDA use.\n        '
        self.vc_config = load_config(vc_config_path)
        self.vc_model = setup_vc_model(config=self.vc_config)
        self.vc_model.load_checkpoint(self.vc_config, vc_checkpoint)
        if use_cuda:
            self.vc_model.cuda()

    def _load_fairseq_from_dir(self, model_dir: str, use_cuda: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Load the fairseq model from a directory.\n\n        We assume it is VITS and the model knows how to load itself from the directory and there is a config.json file in the directory.\n        '
        self.tts_config = VitsConfig()
        self.tts_model = Vits.init_from_config(self.tts_config)
        self.tts_model.load_fairseq_checkpoint(self.tts_config, checkpoint_dir=model_dir, eval=True)
        self.tts_config = self.tts_model.config
        if use_cuda:
            self.tts_model.cuda()

    def _load_tts_from_dir(self, model_dir: str, use_cuda: bool) -> None:
        if False:
            print('Hello World!')
        'Load the TTS model from a directory.\n\n        We assume the model knows how to load itself from the directory and there is a config.json file in the directory.\n        '
        config = load_config(os.path.join(model_dir, 'config.json'))
        self.tts_config = config
        self.tts_model = setup_tts_model(config)
        self.tts_model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
        if use_cuda:
            self.tts_model.cuda()

    def _load_tts(self, tts_checkpoint: str, tts_config_path: str, use_cuda: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Load the TTS model.\n\n        1. Load the model config.\n        2. Init the model from the config.\n        3. Load the model weights.\n        4. Move the model to the GPU if CUDA is enabled.\n        5. Init the speaker manager in the model.\n\n        Args:\n            tts_checkpoint (str): path to the model checkpoint.\n            tts_config_path (str): path to the model config file.\n            use_cuda (bool): enable/disable CUDA use.\n        '
        self.tts_config = load_config(tts_config_path)
        if self.tts_config['use_phonemes'] and self.tts_config['phonemizer'] is None:
            raise ValueError('Phonemizer is not defined in the TTS config.')
        self.tts_model = setup_tts_model(config=self.tts_config)
        if not self.encoder_checkpoint:
            self._set_speaker_encoder_paths_from_tts_config()
        self.tts_model.load_checkpoint(self.tts_config, tts_checkpoint, eval=True)
        if use_cuda:
            self.tts_model.cuda()
        if self.encoder_checkpoint and hasattr(self.tts_model, 'speaker_manager'):
            self.tts_model.speaker_manager.init_encoder(self.encoder_checkpoint, self.encoder_config, use_cuda)

    def _set_speaker_encoder_paths_from_tts_config(self):
        if False:
            while True:
                i = 10
        'Set the encoder paths from the tts model config for models with speaker encoders.'
        if hasattr(self.tts_config, 'model_args') and hasattr(self.tts_config.model_args, 'speaker_encoder_config_path'):
            self.encoder_checkpoint = self.tts_config.model_args.speaker_encoder_model_path
            self.encoder_config = self.tts_config.model_args.speaker_encoder_config_path

    def _load_vocoder(self, model_file: str, model_config: str, use_cuda: bool) -> None:
        if False:
            return 10
        'Load the vocoder model.\n\n        1. Load the vocoder config.\n        2. Init the AudioProcessor for the vocoder.\n        3. Init the vocoder model from the config.\n        4. Move the model to the GPU if CUDA is enabled.\n\n        Args:\n            model_file (str): path to the model checkpoint.\n            model_config (str): path to the model config file.\n            use_cuda (bool): enable/disable CUDA use.\n        '
        self.vocoder_config = load_config(model_config)
        self.vocoder_ap = AudioProcessor(verbose=False, **self.vocoder_config.audio)
        self.vocoder_model = setup_vocoder_model(self.vocoder_config)
        self.vocoder_model.load_checkpoint(self.vocoder_config, model_file, eval=True)
        if use_cuda:
            self.vocoder_model.cuda()

    def split_into_sentences(self, text) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Split give text into sentences.\n\n        Args:\n            text (str): input text in string format.\n\n        Returns:\n            List[str]: list of sentences.\n        '
        return self.seg.segment(text)

    def save_wav(self, wav: List[int], path: str, pipe_out=None) -> None:
        if False:
            return 10
        'Save the waveform as a file.\n\n        Args:\n            wav (List[int]): waveform as a list of values.\n            path (str): output path to save the waveform.\n            pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.\n        '
        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()
        if isinstance(wav, list):
            wav = np.array(wav)
        save_wav(wav=wav, path=path, sample_rate=self.output_sample_rate, pipe_out=pipe_out)

    def voice_conversion(self, source_wav: str, target_wav: str) -> List[int]:
        if False:
            while True:
                i = 10
        output_wav = self.vc_model.voice_conversion(source_wav, target_wav)
        return output_wav

    def tts(self, text: str='', speaker_name: str='', language_name: str='', speaker_wav=None, style_wav=None, style_text=None, reference_wav=None, reference_speaker_name=None, **kwargs) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        '🐸 TTS magic. Run all the models and generate speech.\n\n        Args:\n            text (str): input text.\n            speaker_name (str, optional): speaker id for multi-speaker models. Defaults to "".\n            language_name (str, optional): language id for multi-language models. Defaults to "".\n            speaker_wav (Union[str, List[str]], optional): path to the speaker wav for voice cloning. Defaults to None.\n            style_wav ([type], optional): style waveform for GST. Defaults to None.\n            style_text ([type], optional): transcription of style_wav for Capacitron. Defaults to None.\n            reference_wav ([type], optional): reference waveform for voice conversion. Defaults to None.\n            reference_speaker_name ([type], optional): speaker id of reference waveform. Defaults to None.\n        Returns:\n            List[int]: [description]\n        '
        start_time = time.time()
        wavs = []
        if not text and (not reference_wav):
            raise ValueError('You need to define either `text` (for sythesis) or a `reference_wav` (for voice conversion) to use the Coqui TTS API.')
        if text:
            sens = self.split_into_sentences(text)
            print(' > Text splitted to sentences.')
            print(sens)
        if 'voice_dir' in kwargs:
            self.voice_dir = kwargs['voice_dir']
            kwargs.pop('voice_dir')
        speaker_embedding = None
        speaker_id = None
        if self.tts_speakers_file or hasattr(self.tts_model.speaker_manager, 'name_to_id'):
            if speaker_name and isinstance(speaker_name, str):
                if self.tts_config.use_d_vector_file:
                    speaker_embedding = self.tts_model.speaker_manager.get_mean_embedding(speaker_name, num_samples=None, randomize=False)
                    speaker_embedding = np.array(speaker_embedding)[None, :]
                else:
                    speaker_id = self.tts_model.speaker_manager.name_to_id[speaker_name]
            elif len(self.tts_model.speaker_manager.name_to_id) == 1:
                speaker_id = list(self.tts_model.speaker_manager.name_to_id.values())[0]
            elif not speaker_name and (not speaker_wav):
                raise ValueError(' [!] Looks like you are using a multi-speaker model. You need to define either a `speaker_idx` or a `speaker_wav` to use a multi-speaker model.')
            else:
                speaker_embedding = None
        elif speaker_name and self.voice_dir is None:
            raise ValueError(f' [!] Missing speakers.json file path for selecting speaker {speaker_name}.Define path for speaker.json if it is a multi-speaker model or remove defined speaker idx. ')
        language_id = None
        if self.tts_languages_file or (hasattr(self.tts_model, 'language_manager') and self.tts_model.language_manager is not None):
            if len(self.tts_model.language_manager.name_to_id) == 1:
                language_id = list(self.tts_model.language_manager.name_to_id.values())[0]
            elif language_name and isinstance(language_name, str):
                try:
                    language_id = self.tts_model.language_manager.name_to_id[language_name]
                except KeyError as e:
                    raise ValueError(f' [!] Looks like you use a multi-lingual model. Language {language_name} is not in the available languages: {self.tts_model.language_manager.name_to_id.keys()}.') from e
            elif not language_name:
                raise ValueError(' [!] Look like you use a multi-lingual model. You need to define either a `language_name` or a `style_wav` to use a multi-lingual model.')
            else:
                raise ValueError(f' [!] Missing language_ids.json file path for selecting language {language_name}.Define path for language_ids.json if it is a multi-lingual model or remove defined language idx. ')
        if speaker_wav is not None and self.tts_model.speaker_manager is not None:
            speaker_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(speaker_wav)
        vocoder_device = 'cpu'
        use_gl = self.vocoder_model is None
        if not use_gl:
            vocoder_device = next(self.vocoder_model.parameters()).device
        if self.use_cuda:
            vocoder_device = 'cuda'
        if not reference_wav:
            for sen in sens:
                if hasattr(self.tts_model, 'synthesize'):
                    outputs = self.tts_model.synthesize(text=sen, config=self.tts_config, speaker_id=speaker_name, voice_dirs=self.voice_dir, d_vector=speaker_embedding, speaker_wav=speaker_wav, language=language_name, **kwargs)
                else:
                    outputs = synthesis(model=self.tts_model, text=sen, CONFIG=self.tts_config, use_cuda=self.use_cuda, speaker_id=speaker_id, style_wav=style_wav, style_text=style_text, use_griffin_lim=use_gl, d_vector=speaker_embedding, language_id=language_id)
                waveform = outputs['wav']
                if not use_gl:
                    mel_postnet_spec = outputs['outputs']['model_outputs'][0].detach().cpu().numpy()
                    mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                    vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                    scale_factor = [1, self.vocoder_config['audio']['sample_rate'] / self.tts_model.ap.sample_rate]
                    if scale_factor[1] != 1:
                        print(' > interpolating tts model output.')
                        vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                    else:
                        vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)
                    waveform = self.vocoder_model.inference(vocoder_input.to(vocoder_device))
                if torch.is_tensor(waveform) and waveform.device != torch.device('cpu') and (not use_gl):
                    waveform = waveform.cpu()
                if not use_gl:
                    waveform = waveform.numpy()
                waveform = waveform.squeeze()
                if 'do_trim_silence' in self.tts_config.audio and self.tts_config.audio['do_trim_silence']:
                    waveform = trim_silence(waveform, self.tts_model.ap)
                wavs += list(waveform)
                wavs += [0] * 10000
        else:
            reference_speaker_embedding = None
            reference_speaker_id = None
            if self.tts_speakers_file or hasattr(self.tts_model.speaker_manager, 'name_to_id'):
                if reference_speaker_name and isinstance(reference_speaker_name, str):
                    if self.tts_config.use_d_vector_file:
                        reference_speaker_embedding = self.tts_model.speaker_manager.get_embeddings_by_name(reference_speaker_name)[0]
                        reference_speaker_embedding = np.array(reference_speaker_embedding)[None, :]
                    else:
                        reference_speaker_id = self.tts_model.speaker_manager.name_to_id[reference_speaker_name]
                else:
                    reference_speaker_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(reference_wav)
            outputs = transfer_voice(model=self.tts_model, CONFIG=self.tts_config, use_cuda=self.use_cuda, reference_wav=reference_wav, speaker_id=speaker_id, d_vector=speaker_embedding, use_griffin_lim=use_gl, reference_speaker_id=reference_speaker_id, reference_d_vector=reference_speaker_embedding)
            waveform = outputs
            if not use_gl:
                mel_postnet_spec = outputs[0].detach().cpu().numpy()
                mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                scale_factor = [1, self.vocoder_config['audio']['sample_rate'] / self.tts_model.ap.sample_rate]
                if scale_factor[1] != 1:
                    print(' > interpolating tts model output.')
                    vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)
                waveform = self.vocoder_model.inference(vocoder_input.to(vocoder_device))
            if torch.is_tensor(waveform) and waveform.device != torch.device('cpu'):
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            wavs = waveform.squeeze()
        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio['sample_rate']
        print(f' > Processing time: {process_time}')
        print(f' > Real-time factor: {process_time / audio_time}')
        return wavs