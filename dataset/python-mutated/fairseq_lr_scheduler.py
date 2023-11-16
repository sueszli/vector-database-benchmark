from argparse import Namespace
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.optim import FairseqOptimizer

class FairseqLRScheduler(object):

    def __init__(self, cfg, optimizer):
        if False:
            i = 10
            return i + 15
        super().__init__()
        if optimizer is not None and (not isinstance(optimizer, FairseqOptimizer)):
            raise ValueError('optimizer must be an instance of FairseqOptimizer')
        self.cfg = cfg
        self.optimizer = optimizer
        self.best = None

    @classmethod
    def add_args(cls, parser):
        if False:
            for i in range(10):
                print('nop')
        'Add arguments to the parser for this LR scheduler.'
        dc = getattr(cls, '__dataclass', None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    def state_dict(self):
        if False:
            i = 10
            return i + 15
        'Return the LR scheduler state dict.'
        return {'best': self.best}

    def load_state_dict(self, state_dict):
        if False:
            print('Hello World!')
        'Load an LR scheduler state dict.'
        self.best = state_dict['best']

    def step_begin_epoch(self, epoch):
        if False:
            i = 10
            return i + 15
        'Update the learning rate at the beginning of the given epoch.'
        pass

    def step(self, epoch, val_loss=None):
        if False:
            print('Hello World!')
        'Update the learning rate at the end of the given epoch.'
        if val_loss is not None:
            if self.best is None:
                self.best = val_loss
            else:
                self.best = min(self.best, val_loss)

    def step_update(self, num_updates):
        if False:
            while True:
                i = 10
        'Update the learning rate after each update.'
        return self.optimizer.get_lr()

class LegacyFairseqLRScheduler(FairseqLRScheduler):

    def __init__(self, args: Namespace, optimizer):
        if False:
            i = 10
            return i + 15
        if not isinstance(optimizer, FairseqOptimizer):
            raise ValueError('optimizer must be an instance of FairseqOptimizer')
        self.args = args
        self.optimizer = optimizer
        self.best = None