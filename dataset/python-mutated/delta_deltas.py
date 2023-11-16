import numpy as np
import torch
from fairseq.data.audio.feature_transforms import AudioFeatureTransform, register_audio_feature_transform

@register_audio_feature_transform('delta_deltas')
class DeltaDeltas(AudioFeatureTransform):
    """Expand delta-deltas features from spectrum."""

    @classmethod
    def from_config_dict(cls, config=None):
        if False:
            i = 10
            return i + 15
        _config = {} if config is None else config
        return DeltaDeltas(_config.get('win_length', 5))

    def __init__(self, win_length=5):
        if False:
            i = 10
            return i + 15
        self.win_length = win_length

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.__class__.__name__

    def __call__(self, spectrogram):
        if False:
            return 10
        from torchaudio.functional import compute_deltas
        assert len(spectrogram.shape) == 2, 'spectrogram must be a 2-D tensor.'
        spectrogram = torch.from_numpy(spectrogram).transpose(0, 1)
        delta = compute_deltas(spectrogram)
        delta_delta = compute_deltas(delta)
        out_feat = np.concatenate([spectrogram, delta.numpy(), delta_delta.numpy()], axis=0)
        out_feat = np.transpose(out_feat)
        return out_feat