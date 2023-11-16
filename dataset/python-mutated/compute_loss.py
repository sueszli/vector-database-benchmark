import caffe2.contrib.playground.meter as Meter
from caffe2.python import workspace

class ComputeLoss(Meter.Meter):

    def __init__(self, opts=None, blob_name=''):
        if False:
            return 10
        self.blob_name = blob_name
        self.opts = opts
        self.iter = 0
        self.value = 0

    def Reset(self):
        if False:
            i = 10
            return i + 15
        self.iter = 0
        self.value = 0

    def Add(self):
        if False:
            print('Hello World!')
        'Average values of a blob on each gpu'
        value = 0
        for idx in range(self.opts['distributed']['first_xpu_id'], self.opts['distributed']['first_xpu_id'] + self.opts['distributed']['num_xpus']):
            value += workspace.FetchBlob('{}_{}/{}'.format(self.opts['distributed']['device'], idx, self.blob_name))
        self.value += value
        self.iter += 1

    def Compute(self):
        if False:
            return 10
        result = self.opts['distributed']['num_shards'] * self.value / self.iter
        self.Reset()
        return result