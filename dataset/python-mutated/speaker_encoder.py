from pathlib import Path
from time import perf_counter as timer
from typing import List, Union
import numpy as np
import torch
from torch import nn
from TTS.utils.io import load_fsspec
from TTS.vc.modules.freevc.speaker_encoder import audio
from TTS.vc.modules.freevc.speaker_encoder.hparams import *

class SpeakerEncoder(nn.Module):

    def __init__(self, weights_fpath, device: Union[str, torch.device]=None, verbose=True):
        if False:
            print('Hello World!')
        '\n        :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda").\n        If None, defaults to cuda if it is available on your machine, otherwise the model will\n        run on cpu. Outputs are always returned on the cpu, as numpy arrays.\n        '
        super().__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        start = timer()
        checkpoint = load_fsspec(weights_fpath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state'], strict=False)
        self.to(device)
        if verbose:
            print('Loaded the voice encoder model on %s in %.2f seconds.' % (device.type, timer() - start))

    def forward(self, mels: torch.FloatTensor):
        if False:
            i = 10
            return i + 15
        '\n        Computes the embeddings of a batch of utterance spectrograms.\n        :param mels: a batch of mel spectrograms of same duration as a float32 tensor of shape\n        (batch_size, n_frames, n_channels)\n        :return: the embeddings as a float 32 tensor of shape (batch_size, embedding_size).\n        Embeddings are positive and L2-normed, thus they lay in the range [0, 1].\n        '
        (_, (hidden, _)) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        if False:
            i = 10
            return i + 15
        "\n        Computes where to split an utterance waveform and its corresponding mel spectrogram to\n        obtain partial utterances of <partials_n_frames> each. Both the waveform and the\n        mel spectrogram slices are returned, so as to make each partial utterance waveform\n        correspond to its spectrogram.\n\n        The returned ranges may be indexing further than the length of the waveform. It is\n        recommended that you pad the waveform with zeros up to wav_slices[-1].stop.\n\n        :param n_samples: the number of samples in the waveform\n        :param rate: how many partial utterances should occur per second. Partial utterances must\n        cover the span of the entire utterance, thus the rate should not be lower than the inverse\n        of the duration of a partial utterance. By default, partial utterances are 1.6s long and\n        the minimum rate is thus 0.625.\n        :param min_coverage: when reaching the last partial utterance, it may or may not have\n        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,\n        then the last partial utterance will be considered by zero-padding the audio. Otherwise,\n        it will be discarded. If there aren't enough frames for one partial utterance,\n        this parameter is ignored so that the function always returns at least one slice.\n        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index\n        respectively the waveform and the mel spectrogram with these slices to obtain the partial\n        utterances.\n        "
        assert 0 < min_coverage <= 1
        samples_per_frame = int(sampling_rate * mel_window_step / 1000)
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round(sampling_rate / rate / samples_per_frame))
        assert 0 < frame_step, 'The rate is too high'
        assert frame_step <= partials_n_frames, 'The rate is too low, it should be %f at least' % (sampling_rate / (samples_per_frame * partials_n_frames))
        (wav_slices, mel_slices) = ([], [])
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]
        return (wav_slices, mel_slices)

    def embed_utterance(self, wav: np.ndarray, return_partials=False, rate=1.3, min_coverage=0.75):
        if False:
            return 10
        "\n        Computes an embedding for a single utterance. The utterance is divided in partial\n        utterances and an embedding is computed for each. The complete utterance embedding is the\n        L2-normed average embedding of the partial utterances.\n\n        TODO: independent batched version of this function\n\n        :param wav: a preprocessed utterance waveform as a numpy array of float32\n        :param return_partials: if True, the partial embeddings will also be returned along with\n        the wav slices corresponding to each partial utterance.\n        :param rate: how many partial utterances should occur per second. Partial utterances must\n        cover the span of the entire utterance, thus the rate should not be lower than the inverse\n        of the duration of a partial utterance. By default, partial utterances are 1.6s long and\n        the minimum rate is thus 0.625.\n        :param min_coverage: when reaching the last partial utterance, it may or may not have\n        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,\n        then the last partial utterance will be considered by zero-padding the audio. Otherwise,\n        it will be discarded. If there aren't enough frames for one partial utterance,\n        this parameter is ignored so that the function always returns at least one slice.\n        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If\n        <return_partials> is True, the partial utterances as a numpy array of float32 of shape\n        (n_partials, model_embedding_size) and the wav partials as a list of slices will also be\n        returned.\n        "
        (wav_slices, mel_slices) = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), 'constant')
        mel = audio.wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        with torch.no_grad():
            mels = torch.from_numpy(mels).to(self.device)
            partial_embeds = self(mels).cpu().numpy()
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)
        if return_partials:
            return (embed, partial_embeds, wav_slices)
        return embed

    def embed_speaker(self, wavs: List[np.ndarray], **kwargs):
        if False:
            while True:
                i = 10
        '\n        Compute the embedding of a collection of wavs (presumably from the same speaker) by\n        averaging their embedding and L2-normalizing it.\n\n        :param wavs: list of wavs a numpy arrays of float32.\n        :param kwargs: extra arguments to embed_utterance()\n        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,).\n        '
        raw_embed = np.mean([self.embed_utterance(wav, return_partials=False, **kwargs) for wav in wavs], axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)