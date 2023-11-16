import time
from modelscope.metainfo import Hooks
from modelscope.utils.constant import LogKeys
from .builder import HOOKS
from .hook import Hook
from .priority import Priority

@HOOKS.register_module(module_name=Hooks.IterTimerHook)
class IterTimerHook(Hook):
    PRIORITY = Priority.LOW

    def before_epoch(self, trainer):
        if False:
            for i in range(10):
                print('nop')
        self.start_time = time.time()

    def before_iter(self, trainer):
        if False:
            return 10
        trainer.log_buffer.update({LogKeys.DATA_LOAD_TIME: time.time() - self.start_time})

    def after_iter(self, trainer):
        if False:
            i = 10
            return i + 15
        trainer.log_buffer.update({LogKeys.ITER_TIME: time.time() - self.start_time})
        self.start_time = time.time()