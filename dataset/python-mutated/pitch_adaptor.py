from typing import Callable, Tuple
import torch
import torch.nn as nn
from TTS.tts.layers.delightful_tts.variance_predictor import VariancePredictor
from TTS.tts.utils.helpers import average_over_durations

class PitchAdaptor(nn.Module):
    """Module to get pitch embeddings via pitch predictor

    Args:
        n_input (int): Number of pitch predictor input channels.
        n_hidden (int): Number of pitch predictor hidden channels.
        n_out (int): Number of pitch predictor out channels.
        kernel size (int): Size of the kernel for conv layers.
        emb_kernel_size (int): Size the kernel for the pitch embedding.
        p_dropout (float): Probability of dropout.
        lrelu_slope (float): Slope for the leaky relu.

    Inputs: inputs, mask
        - **inputs** (batch, time1, dim): Tensor containing input vector
        - **target** (batch, 1, time2): Tensor containing the pitch target
        - **dr** (batch, time1): Tensor containing aligner durations vector
        - **mask** (batch, time1): Tensor containing indices to be masked
    Returns:
        - **pitch prediction** (batch, 1, time1): Tensor produced by pitch predictor
        - **pitch embedding** (batch, channels, time1): Tensor produced pitch pitch adaptor
        - **average pitch target(train only)** (batch, 1, time1): Tensor produced after averaging over durations
    """

    def __init__(self, n_input: int, n_hidden: int, n_out: int, kernel_size: int, emb_kernel_size: int, p_dropout: float, lrelu_slope: float):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.pitch_predictor = VariancePredictor(channels_in=n_input, channels=n_hidden, channels_out=n_out, kernel_size=kernel_size, p_dropout=p_dropout, lrelu_slope=lrelu_slope)
        self.pitch_emb = nn.Conv1d(1, n_input, kernel_size=emb_kernel_size, padding=int((emb_kernel_size - 1) / 2))

    def get_pitch_embedding_train(self, x: torch.Tensor, target: torch.Tensor, dr: torch.IntTensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Shapes:\n            x: :math: `[B, T_src, C]`\n            target: :math: `[B, 1, T_max2]`\n            dr: :math: `[B, T_src]`\n            mask: :math: `[B, T_src]`\n        '
        pitch_pred = self.pitch_predictor(x, mask)
        pitch_pred.unsqueeze_(1)
        avg_pitch_target = average_over_durations(target, dr)
        pitch_emb = self.pitch_emb(avg_pitch_target)
        return (pitch_pred, avg_pitch_target, pitch_emb)

    def get_pitch_embedding(self, x: torch.Tensor, mask: torch.Tensor, pitch_transform: Callable, pitch_mean: torch.Tensor, pitch_std: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        pitch_pred = self.pitch_predictor(x, mask)
        if pitch_transform is not None:
            pitch_pred = pitch_transform(pitch_pred, (~mask).sum(), pitch_mean, pitch_std)
        pitch_pred.unsqueeze_(1)
        pitch_emb_pred = self.pitch_emb(pitch_pred)
        return (pitch_emb_pred, pitch_pred)