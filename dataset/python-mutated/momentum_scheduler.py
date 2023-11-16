import torch
from allennlp.common.registrable import Registrable
from allennlp.training.scheduler import Scheduler

class MomentumScheduler(Scheduler, Registrable):

    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int=-1) -> None:
        if False:
            print('Hello World!')
        super().__init__(optimizer, 'momentum', last_epoch)

    def get_values(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError