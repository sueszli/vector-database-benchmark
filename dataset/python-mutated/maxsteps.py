from bigdl.orca.learn.pytorch.callbacks import Callback
import math

class MaxstepsCallback(Callback):

    def __init__(self, max_step) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.max_step = max_step

    def before_run(self, runner):
        if False:
            return 10
        runner.num_epochs = math.ceil(self.max_step / len(runner.train_loader))

    def after_train_iter(self, runner):
        if False:
            return 10
        if runner.global_step >= self.max_step:
            runner.stop = True