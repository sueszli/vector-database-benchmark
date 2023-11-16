import torch
from .MetricBase import MetricBase

class CUDAMetric(MetricBase):

    def __init__(self, rank: int, name: str):
        if False:
            while True:
                i = 10
        self.rank = rank
        self.name = name
        self.start = None
        self.end = None

    def record_start(self):
        if False:
            while True:
                i = 10
        self.start = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(self.rank):
            self.start.record()

    def record_end(self):
        if False:
            return 10
        self.end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(self.rank):
            self.end.record()

    def elapsed_time(self):
        if False:
            while True:
                i = 10
        if not self.start.query():
            raise RuntimeError('start event did not complete')
        if not self.end.query():
            raise RuntimeError('end event did not complete')
        return self.start.elapsed_time(self.end)

    def synchronize(self):
        if False:
            i = 10
            return i + 15
        self.start.synchronize()
        self.end.synchronize()