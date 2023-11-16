from modelscope.metainfo import Hooks
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.hook import Hook
from modelscope.trainers.hooks.priority import Priority
from modelscope.utils.constant import DistributedParallelType
from modelscope.utils.device import create_device
from modelscope.utils.torch_utils import get_local_rank, init_dist

@HOOKS.register_module(module_name=Hooks.DDPHook)
class DDPHook(Hook):
    PRIORITY = Priority.LOW

    def __init__(self, launcher):
        if False:
            while True:
                i = 10
        "The DDP Hook for data parallel\n\n        Args:\n            launcher(str, required): The launcher info, can be 'pytorch' or 'mpi' or 'slurm'\n        "
        assert launcher is not None
        self.launcher = launcher
        self.wrapped = False

    def after_init(self, trainer):
        if False:
            for i in range(10):
                print('nop')
        init_dist(self.launcher)
        local_rank = get_local_rank()
        trainer.device = create_device(f'cuda:{local_rank}')
        trainer.model.to(trainer.device)
        trainer.parallel_groups[DistributedParallelType.DP] = None

    def before_run(self, trainer):
        if False:
            while True:
                i = 10
        self.wrap_module(trainer)

    def before_val(self, trainer):
        if False:
            print('Hello World!')
        self.wrap_module(trainer)

    def wrap_module(self, trainer):
        if False:
            return 10
        if not self.wrapped:
            trainer.model = trainer.to_parallel(trainer.model)
            self.wrapped = True