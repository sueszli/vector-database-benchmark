from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
import logging as _logging
from coremltools.converters._profile_utils import _profile
from tqdm import tqdm as _tqdm

@_profile
def common_pass(prog):
    if False:
        for i in range(10):
            print('nop')
    passes = ['common::const_elimination', 'common::divide_to_multiply', 'common::const_elimination', 'common::loop_invariant_elimination', 'common::remove_symbolic_reshape', 'common::noop_elimination', 'common::fuse_matmul_weight_bias', 'common::fuse_gelu_tanh_approximation', 'common::reduce_transposes', 'common::fuse_bias_conv', 'common::fuse_elementwise_to_batchnorm', 'common::fuse_onehot_matmul_to_gather', 'common::fuse_layernorm_or_instancenorm', 'common::dead_code_elimination']
    _logging.debug('Program before common passes:\n{}'.format(prog))
    prog.validate()
    for p in _tqdm(passes, desc='Running MIL optimization passes', unit=' passes'):
        _logging.info('Performing pass: "{}"'.format(p))
        PASS_REGISTRY[p](prog)
        prog.validate()
    _logging.debug('Program after common passes:\n{}'.format(prog))