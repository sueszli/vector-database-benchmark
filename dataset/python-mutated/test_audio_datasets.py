import itertools
import unittest
import numpy as np
from parameterized import parameterized
import paddle

def parameterize(*params):
    if False:
        while True:
            i = 10
    return parameterized.expand(list(itertools.product(*params)))

class TestAudioDatasets(unittest.TestCase):

    @parameterize(['dev', 'train'], [40, 64])
    def test_tess_dataset(self, mode: str, params: int):
        if False:
            print('Hello World!')
        '\n        TESS dataset\n        Reference:\n            Toronto emotional speech set (TESS) https://tspace.library.utoronto.ca/handle/1807/24487\n            https://doi.org/10.5683/SP2/E8H2MF\n        '
        archive = {'url': 'https://bj.bcebos.com/paddleaudio/datasets/TESS_Toronto_emotional_speech_set_lite.zip', 'md5': '9ffb5e3adf28d4d6b787fa94bd59b975'}
        tess_dataset = paddle.audio.datasets.TESS(mode=mode, feat_type='mfcc', n_mfcc=params, archive=archive)
        idx = np.random.randint(0, 30)
        elem = tess_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params)
        self.assertTrue(0 <= elem[1] <= 6)
        tess_dataset = paddle.audio.datasets.TESS(mode=mode, feat_type='spectrogram', n_fft=params)
        elem = tess_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params // 2 + 1)
        self.assertTrue(0 <= elem[1] <= 6)
        tess_dataset = paddle.audio.datasets.TESS(mode='dev', feat_type='logmelspectrogram', n_mels=params)
        elem = tess_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params)
        self.assertTrue(0 <= elem[1] <= 6)
        tess_dataset = paddle.audio.datasets.TESS(mode='dev', feat_type='melspectrogram', n_mels=params)
        elem = tess_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params)
        self.assertTrue(0 <= elem[1] <= 6)

    @parameterize(['dev', 'train'], [40, 64])
    def test_esc50_dataset(self, mode: str, params: int):
        if False:
            i = 10
            return i + 15
        '\n        ESC50 dataset\n        Reference:\n            ESC: Dataset for Environmental Sound Classification\n            http://dx.doi.org/10.1145/2733373.2806390\n        '
        archive = {'url': 'https://bj.bcebos.com/paddleaudio/datasets/ESC-50-master-lite.zip', 'md5': '1e9ba53265143df5b2804a743f2d1956'}
        esc50_dataset = paddle.audio.datasets.ESC50(mode=mode, feat_type='raw', archive=archive)
        idx = np.random.randint(0, 6)
        elem = esc50_dataset[idx]
        self.assertTrue(elem[0].shape[0] == 220500)
        self.assertTrue(0 <= elem[1] <= 2)
        esc50_dataset = paddle.audio.datasets.ESC50(mode=mode, feat_type='mfcc', n_mfcc=params, archive=archive)
        idx = np.random.randint(0, 6)
        elem = esc50_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params)
        self.assertTrue(0 <= elem[1] <= 2)
        esc50_dataset = paddle.audio.datasets.ESC50(mode=mode, feat_type='spectrogram', n_fft=params)
        elem = esc50_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params // 2 + 1)
        self.assertTrue(0 <= elem[1] <= 2)
        esc50_dataset = paddle.audio.datasets.ESC50(mode=mode, feat_type='logmelspectrogram', n_mels=params)
        elem = esc50_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params)
        self.assertTrue(0 <= elem[1] <= 2)
        esc50_dataset = paddle.audio.datasets.ESC50(mode=mode, feat_type='melspectrogram', n_mels=params)
        elem = esc50_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params)
        self.assertTrue(0 <= elem[1] <= 2)
if __name__ == '__main__':
    unittest.main()