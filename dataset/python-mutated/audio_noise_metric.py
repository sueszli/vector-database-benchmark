from typing import Dict
from modelscope.metainfo import Metrics
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS, MetricKeys
from modelscope.utils.registry import default_group

@METRICS.register_module(group_key=default_group, module_name=Metrics.audio_noise_metric)
class AudioNoiseMetric(Metric):
    """
    The metric computation class for acoustic noise suppression task.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.loss = []
        self.amp_loss = []
        self.phase_loss = []
        self.sisnr = []

    def add(self, outputs: Dict, inputs: Dict):
        if False:
            return 10
        self.loss.append(outputs['loss'].data.cpu())
        self.amp_loss.append(outputs['amp_loss'].data.cpu())
        self.phase_loss.append(outputs['phase_loss'].data.cpu())
        self.sisnr.append(outputs['sisnr'].data.cpu())

    def evaluate(self):
        if False:
            while True:
                i = 10
        avg_loss = sum(self.loss) / len(self.loss)
        avg_sisnr = sum(self.sisnr) / len(self.sisnr)
        avg_amp = sum(self.amp_loss) / len(self.amp_loss)
        avg_phase = sum(self.phase_loss) / len(self.phase_loss)
        total_loss = avg_loss + avg_amp + avg_phase + avg_sisnr
        return {'total_loss': total_loss.item(), 'avg_sisnr': -avg_sisnr.item(), MetricKeys.AVERAGE_LOSS: avg_loss.item()}

    def merge(self, other: 'AudioNoiseMetric'):
        if False:
            while True:
                i = 10
        self.loss.extend(other.loss)
        self.amp_loss.extend(other.amp_loss)
        self.phase_loss.extend(other.phase_loss)
        self.sisnr.extend(other.sisnr)

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.loss, self.amp_loss, self.phase_loss, self.sisnr)

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        (self.loss, self.amp_loss, self.phase_loss, self.sisnr) = state