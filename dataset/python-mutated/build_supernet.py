from copy import deepcopy
from lib.utils.builder_util import modify_block_args
from lib.models.blocks import get_Bottleneck, InvertedResidual
from timm.models.efficientnet_blocks import *
from nni.nas.pytorch import mutables

class SuperNetBuilder:
    """ Build Trunk Blocks
    """

    def __init__(self, choices, channel_multiplier=1.0, channel_divisor=8, channel_min=None, output_stride=32, pad_type='', act_layer=None, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0.0, feature_location='', verbose=False, resunit=False, dil_conv=False, logger=None):
        if False:
            i = 10
            return i + 15
        self.choices = [[x, y] for x in choices['kernel_size'] for y in choices['exp_ratio']]
        self.choices_num = len(self.choices) - 1
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.act_layer = act_layer
        self.se_kwargs = se_kwargs
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.drop_path_rate = drop_path_rate
        self.feature_location = feature_location
        assert feature_location in ('pre_pwl', 'post_exp', '')
        self.verbose = verbose
        self.resunit = resunit
        self.dil_conv = dil_conv
        self.logger = logger
        self.in_chs = None

    def _round_channels(self, chs):
        if False:
            i = 10
            return i + 15
        return round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    def _make_block(self, ba, choice_idx, block_idx, block_count, resunit=False, dil_conv=False):
        if False:
            return 10
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self._round_channels(ba['out_chs'])
        if 'fake_in_chs' in ba and ba['fake_in_chs']:
            ba['fake_in_chs'] = self._round_channels(ba['fake_in_chs'])
        ba['norm_layer'] = self.norm_layer
        ba['norm_kwargs'] = self.norm_kwargs
        ba['pad_type'] = self.pad_type
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'] is not None else self.act_layer
        assert ba['act_layer'] is not None
        if bt == 'ir':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            if self.verbose:
                self.logger.info('  InvertedResidual {}, Args: {}'.format(block_idx, str(ba)))
            block = InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            ba['drop_path_rate'] = drop_path_rate
            ba['se_kwargs'] = self.se_kwargs
            if self.verbose:
                self.logger.info('  DepthwiseSeparable {}, Args: {}'.format(block_idx, str(ba)))
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'cn':
            if self.verbose:
                self.logger.info('  ConvBnAct {}, Args: {}'.format(block_idx, str(ba)))
            block = ConvBnAct(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        if choice_idx == self.choice_num - 1:
            self.in_chs = ba['out_chs']
        return block

    def __call__(self, in_chs, model_block_args):
        if False:
            for i in range(10):
                print('nop')
        ' Build the blocks\n        Args:\n            in_chs: Number of input-channels passed to first block\n            model_block_args: A list of lists, outer list defines stages, inner\n                list contains strings defining block configuration(s)\n        Return:\n             List of block stacks (each stack wrapped in nn.Sequential)\n        '
        if self.verbose:
            logging.info('Building model trunk with %d stages...' % len(model_block_args))
        self.in_chs = in_chs
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        feature_idx = 0
        stages = []
        for (stage_idx, stage_block_args) in enumerate(model_block_args):
            last_stack = stage_idx == len(model_block_args) - 1
            if self.verbose:
                self.logger.info('Stack: {}'.format(stage_idx))
            assert isinstance(stage_block_args, list)
            for (block_idx, block_args) in enumerate(stage_block_args):
                last_block = block_idx == len(stage_block_args) - 1
                if self.verbose:
                    self.logger.info(' Block: {}'.format(block_idx))
                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:
                    block_args['stride'] = 1
                next_dilation = current_dilation
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                    else:
                        current_stride = next_output_stride
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation
                if stage_idx == 0 or stage_idx == 6:
                    self.choice_num = 1
                else:
                    self.choice_num = len(self.choices)
                    if self.dil_conv:
                        self.choice_num += 2
                choice_blocks = []
                block_args_copy = deepcopy(block_args)
                if self.choice_num == 1:
                    block = self._make_block(block_args, 0, total_block_idx, total_block_count)
                    choice_blocks.append(block)
                else:
                    for (choice_idx, choice) in enumerate(self.choices):
                        block_args = deepcopy(block_args_copy)
                        block_args = modify_block_args(block_args, choice[0], choice[1])
                        block = self._make_block(block_args, choice_idx, total_block_idx, total_block_count)
                        choice_blocks.append(block)
                    if self.dil_conv:
                        block_args = deepcopy(block_args_copy)
                        block_args = modify_block_args(block_args, 3, 0)
                        block = self._make_block(block_args, self.choice_num - 2, total_block_idx, total_block_count, resunit=self.resunit, dil_conv=self.dil_conv)
                        choice_blocks.append(block)
                        block_args = deepcopy(block_args_copy)
                        block_args = modify_block_args(block_args, 5, 0)
                        block = self._make_block(block_args, self.choice_num - 1, total_block_idx, total_block_count, resunit=self.resunit, dil_conv=self.dil_conv)
                        choice_blocks.append(block)
                    if self.resunit:
                        block = get_Bottleneck(block.conv_pw.in_channels, block.conv_pwl.out_channels, block.conv_dw.stride[0])
                        choice_blocks.append(block)
                choice_block = mutables.LayerChoice(choice_blocks)
                stages.append(choice_block)
                total_block_idx += 1
        return stages