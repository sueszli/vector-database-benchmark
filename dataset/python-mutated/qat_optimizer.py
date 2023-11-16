import copy
import paddle
from paddle.static.quantization.quanter import _quant_config_default, quant_aware
from .meta_optimizer_base import MetaOptimizerBase

class QATOptimizer(MetaOptimizerBase):

    def __init__(self, optimizer):
        if False:
            return 10
        super().__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = ['AMPOptimizer', 'LarsOptimizer', 'LambOptimizer', 'GraphExecutionOptimizer', 'RecomputeOptimizer', 'GradientMergeOptimizer']
        self.meta_optimizers_black_list = []

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer, user_defined_strategy):
        if False:
            while True:
                i = 10
        super()._set_basic_info(loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _can_apply(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.role_maker._is_collective:
            return False
        if self.user_defined_strategy.qat:
            return True
        return False

    def _disable_strategy(self, dist_strategy):
        if False:
            i = 10
            return i + 15
        dist_strategy.qat = False
        dist_strategy.qat_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        if False:
            return 10
        dist_strategy.qat = True
        dist_strategy.qat_configs = {'channel_wise_abs_max': True, 'weight_bits': 8, 'activation_bits': 8, 'not_quant_pattern': [], 'algo': ''}

    def _gen_qat_config(self):
        if False:
            i = 10
            return i + 15
        config = self.user_defined_strategy.qat_configs
        qat_config = copy.deepcopy(_quant_config_default)
        qat_config['quantize_op_types'] = ['conv2d', 'depthwise_conv2d', 'mul', 'matmul', 'matmul_v2']
        qat_config['weight_quantize_type'] = 'channel_wise_abs_max' if config['channel_wise_abs_max'] else 'abs_max'
        qat_config['weight_bits'] = config['weight_bits']
        qat_config['activation_bits'] = config['activation_bits']
        qat_config['not_quant_pattern'] = list(config['not_quant_pattern'])
        return qat_config

    def _replace_program(self, main_program, refer_program):
        if False:
            while True:
                i = 10
        main_program._rebuild_from_desc(refer_program.desc)

    def minimize_impl(self, loss, startup_program=None, parameter_list=None, no_grad_set=None):
        if False:
            while True:
                i = 10
        (optimize_ops, params_grads) = self.inner_opt.minimize(loss, startup_program, parameter_list, no_grad_set)
        device = paddle.device.get_device()
        place = paddle.set_device(device)
        qat_config = self._gen_qat_config()
        qat_program = quant_aware(loss.block.program, place, config=qat_config, return_program=True)
        self._replace_program(loss.block.program, qat_program)
        return (optimize_ops, params_grads)

    def qat_init(self, place, scope=None, test_program=None):
        if False:
            for i in range(10):
                print('nop')
        if test_program is not None:
            qat_config = self._gen_qat_config()
            qat_program = quant_aware(test_program, place, scope=scope, config=qat_config, for_test=True, return_program=True)
            self._replace_program(test_program, qat_program)