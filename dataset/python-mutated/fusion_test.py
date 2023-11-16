import logging
import os
from test.cinn.passes.pass_test import PassTest
logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(name='pass_test')

class FusionTest(PassTest):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)

    def init_input_data(self):
        if False:
            return 10
        'Set feed data'
        self.feed_data = {}
        logger.warn('No Input Data')

    def build_program(self, builder, target):
        if False:
            while True:
                i = 10
        ' '
        raise Exception('Not implemented.')

    def check_fusion_outputs(self, group_size, max_relative_error=1e-05, all_equal=False, equal_nan=False):
        if False:
            return 10
        base_passes = ['AutoCast', 'Decomposer', 'TransToCustomCallPass']
        fusion_passes = ['OpFusionPass', 'FusionMergePass']
        real_group_size = self.get_pass_size(base_passes + fusion_passes)
        logger.debug(f'The model has been fused into {real_group_size} groups')
        self.assertEqual(real_group_size, group_size, msg='The model should be fused into {} groups, but actually fused {} groups'.format(group_size, real_group_size))
        cinn_no_fusion_outputs = self.get_pass_outputs(base_passes)
        cinn_fusion_outputs = self.get_pass_outputs(base_passes + fusion_passes)
        logger.debug('============ Check Outputs ============')
        self.check_results(cinn_no_fusion_outputs, cinn_fusion_outputs, max_relative_error, all_equal, equal_nan)