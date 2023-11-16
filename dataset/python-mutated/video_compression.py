"""
This module implements a wrapper for video compression defence with FFmpeg.

| Please keep in mind the limitations of defences. For details on how to evaluate classifier security in general,
    see https://arxiv.org/abs/1902.06705.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
from tempfile import TemporaryDirectory
from typing import Optional, Tuple
import warnings
import numpy as np
from tqdm.auto import tqdm
from art import config
from art.defences.preprocessor.preprocessor import Preprocessor
logger = logging.getLogger(__name__)

class VideoCompression(Preprocessor):
    """
    Implement FFmpeg wrapper for video compression defence based on H.264/MPEG-4 AVC.

    Video compression uses H.264 video encoding. The video quality is controlled with the constant rate factor
    parameter. More information on the constant rate factor: https://trac.ffmpeg.org/wiki/Encode/H.264.
    """
    params = ['video_format', 'constant_rate_factor', 'channels_first', 'verbose']

    def __init__(self, *, video_format: str, constant_rate_factor: int=28, channels_first: bool=False, apply_fit: bool=False, apply_predict: bool=True, verbose: bool=False):
        if False:
            while True:
                i = 10
        '\n        Create an instance of VideoCompression.\n\n        :param video_format: Specify one of supported video file extensions, e.g. `avi`, `mp4` or `mkv`.\n        :param constant_rate_factor: Specify constant rate factor (range 0 to 51, where 0 is lossless).\n        :param channels_first: Set channels first or last.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        :param verbose: Show progress bars.\n        '
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.video_format = video_format
        self.constant_rate_factor = constant_rate_factor
        self.channels_first = channels_first
        self.verbose = verbose
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray]=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if False:
            return 10
        '\n        Apply video compression to sample `x`.\n\n        :param x: Sample to compress of shape NCFHW or NFHWC. `x` values are expected to be either in range [0, 1] or\n                  [0, 255].\n        :param y: Labels of the sample `x`. This function does not affect them in any way.\n        :return: Compressed sample.\n        '

        def compress_video(x: np.ndarray, video_format: str, constant_rate_factor: int, dir_: str=''):
            if False:
                while True:
                    i = 10
            '\n            Apply video compression to video input of shape (frames, height, width, channel).\n            '
            import ffmpeg
            video_path = os.path.join(dir_, f'tmp_video.{video_format}')
            (_, height, width, _) = x.shape
            if height % 2 != 0 or width % 2 != 0:
                warnings.warn('Codec might require even number of pixels in height and width.')
            process = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}').output(video_path, pix_fmt='yuv420p', vcodec='libx264', crf=constant_rate_factor).overwrite_output().run_async(pipe_stdin=True, quiet=True)
            process.stdin.write(x.flatten().astype(np.uint8).tobytes())
            process.stdin.close()
            process.wait()
            (stdout, _) = ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True)
            return np.frombuffer(stdout, np.uint8).reshape(x.shape)
        if x.ndim != 5:
            raise ValueError('Video compression can only be applied to spatio-temporal data.')
        if self.channels_first:
            x = np.transpose(x, (0, 2, 3, 4, 1))
        scale = 1
        if x.min() >= 0 and x.max() <= 1.0:
            scale = 255
        x_compressed = x.copy()
        with TemporaryDirectory(dir=config.ART_DATA_PATH) as tmp_dir:
            for (i, x_i) in enumerate(tqdm(x, desc='Video compression', disable=not self.verbose)):
                x_i *= scale
                x_compressed[i] = compress_video(x_i, self.video_format, self.constant_rate_factor, dir_=tmp_dir)
        x_compressed = x_compressed / scale
        x_compressed = x_compressed.astype(x.dtype)
        if self.channels_first:
            x_compressed = np.transpose(x_compressed, (0, 4, 1, 2, 3))
        return (x_compressed, y)

    def _check_params(self) -> None:
        if False:
            print('Hello World!')
        if not (isinstance(self.constant_rate_factor, int) and 0 <= self.constant_rate_factor < 52):
            raise ValueError('Constant rate factor must be an integer in the range [0, 51].')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')