"""
This module implements a wrapper for video compression defence with FFmpeg.

| Please keep in mind the limitations of defences. For details on how to evaluate classifier security in general,
    see https://arxiv.org/abs/1902.06705.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, TYPE_CHECKING
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from art.defences.preprocessor.video_compression import VideoCompression
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    import torch

class VideoCompressionPyTorch(PreprocessorPyTorch):
    """
    Implement FFmpeg wrapper for video compression defence based on H.264/MPEG-4 AVC.

    Video compression uses H.264 video encoding. The video quality is controlled with the constant rate factor
    parameter. More information on the constant rate factor: https://trac.ffmpeg.org/wiki/Encode/H.264.
    """
    params = ['video_format', 'constant_rate_factor', 'channels_first', 'verbose']

    def __init__(self, *, video_format: str, constant_rate_factor: int=28, channels_first: bool=False, apply_fit: bool=False, apply_predict: bool=True, device_type: str='gpu', verbose: bool=False):
        if False:
            print('Hello World!')
        '\n        Create an instance of VideoCompression.\n\n        :param video_format: Specify one of supported video file extensions, e.g. `avi`, `mp4` or `mkv`.\n        :param constant_rate_factor: Specify constant rate factor (range 0 to 51, where 0 is lossless).\n        :param channels_first: Set channels first or last.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.\n        :param verbose: Show progress bars.\n        '
        from torch.autograd import Function
        super().__init__(device_type=device_type, is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.video_format = video_format
        self.constant_rate_factor = constant_rate_factor
        self.channels_first = channels_first
        self.verbose = verbose
        self._check_params()
        self.compression_numpy = VideoCompression(video_format=video_format, constant_rate_factor=constant_rate_factor, channels_first=channels_first, apply_fit=apply_fit, apply_predict=apply_predict, verbose=verbose)

        class CompressionPyTorchNumpy(Function):
            """
            Function running Preprocessor.
            """

            @staticmethod
            def forward(ctx, input):
                if False:
                    while True:
                        i = 10
                numpy_input = input.detach().cpu().numpy()
                (result, _) = self.compression_numpy(numpy_input)
                return input.new(result)

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    i = 10
                    return i + 15
                numpy_go = grad_output.cpu().numpy()
                result = self.compression_numpy.estimate_gradient(None, numpy_go)
                return grad_output.new(result)
        self._compression_pytorch_numpy = CompressionPyTorchNumpy

    def forward(self, x: 'torch.Tensor', y: Optional['torch.Tensor']=None) -> Tuple['torch.Tensor', Optional['torch.Tensor']]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply video compression to sample `x`.\n\n        :param x: Sample to compress of shape NCFHW or NFHWC. `x` values are expected to be either in range [0, 1] or\n                  [0, 255].\n        :param y: Labels of the sample `x`. This function does not affect them in any way.\n        :return: Compressed sample.\n        '
        scale = 1
        if x.min() >= 0 and x.max() <= 1.0:
            scale = 255
        x = x * scale
        x_compressed = self._compression_pytorch_numpy.apply(x)
        x_compressed = x_compressed / scale
        return (x_compressed, y)

    def _check_params(self) -> None:
        if False:
            return 10
        if not (isinstance(self.constant_rate_factor, int) and 0 <= self.constant_rate_factor < 52):
            raise ValueError('Constant rate factor must be an integer in the range [0, 51].')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')