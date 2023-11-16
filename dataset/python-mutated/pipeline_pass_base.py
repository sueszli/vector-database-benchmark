import logging
import paddle
from paddle.base import core
from ..utils.log_utils import get_logger
from .pass_base import PassBase
from .pass_utils import set_skip_gc_vars
logger = get_logger(logging.INFO)

class PipelinePassBase(PassBase):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def _check_self(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def _check_conflict(self, other_pass):
        if False:
            for i in range(10):
                print('nop')
        return True

    def _create_job_list(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        An interface that MUST be implemented by subclasses.\n        '
        pass

    def _partial_programs(self, program):
        if False:
            print('Hello World!')
        '\n        An interface that MUST be implemented by subclasses.\n        The return value MUST be two lists, one is a list of types(str), another\n        is a list of sub programs.\n        For example:\n        return [FORWARD, BACKWARD, OPT], [fwd_prog, bwd_prog, opt_prog]\n        or\n        return [FORWARD], [fwd_prog]\n        '
        pass

    def _apply_single_impl(self, main_program, startup_program, context):
        if False:
            return 10
        "\n        The shared process is implemented in this function and new subclass only need\n        to implement two interfaces above, 'create_job_list' and 'partial_programs'.\n        "
        (job_types, sub_programs) = self._partial_programs(main_program)
        for i in range(len(job_types)):
            logger.debug(f'sub_program type: {job_types[i]}, sum_program:\n{sub_programs[i]}')
        jobs = self._create_job_list()
        type_to_program = set_skip_gc_vars(self.get_attr('num_micro_batches'), job_types, sub_programs, jobs)
        for type in type_to_program.keys():
            if paddle.framework.get_flags('FLAGS_enable_pir_in_executor')['FLAGS_enable_pir_in_executor']:
                type_to_program[type] = paddle.pir.translate_to_pir(type_to_program[type].desc)
            else:
                type_to_program[type] = type_to_program[type].desc
        plan = core.Plan(jobs, type_to_program)
        context.set_attr('plan', plan)