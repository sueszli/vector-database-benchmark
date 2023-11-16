from dataclasses import dataclass
from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler

@dataclass
class PassThroughScheduleConfig(FairseqDataclass):
    pass

@register_lr_scheduler('pass_through', dataclass=PassThroughScheduleConfig)
class PassThroughScheduleSchedule(FairseqLRScheduler):
    """Delegate lr scheduling to the optimizer."""

    def __init__(self, cfg: PassThroughScheduleConfig, optimizer):
        if False:
            return 10
        super().__init__(cfg, optimizer)
        assert hasattr(optimizer, 'lr_scheduler') and optimizer.lr_scheduler is not None, 'Pass-through schedule can only be used with optimizers with their own schedulers'

    def state_dict(self):
        if False:
            while True:
                i = 10
        return self.optimizer.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        if False:
            for i in range(10):
                print('nop')
        self.optimizer.lr_scheduler.load_state_dict(state_dict)

    def step_begin_epoch(self, epoch):
        if False:
            while True:
                i = 10
        'Update the learning rate at the beginning of the given epoch.'
        return self.optimizer.lr_scheduler.step_begin_epoch(epoch)

    def step_update(self, num_updates):
        if False:
            print('Hello World!')
        'Update the learning rate after each update.'
        return self.optimizer.lr_scheduler.step_update(num_updates)