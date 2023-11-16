from lib.utils.util import *
from timm.models.efficientnet_blocks import *

class ChildNetBuilder:

    def __init__(self, channel_multiplier=1.0, channel_divisor=8, channel_min=None, output_stride=32, pad_type='', act_layer=None, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_path_rate=0.0, feature_location='', verbose=False, logger=None):
        if False:
            print('Hello World!')
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
        self.in_chs = None
        self.features = OrderedDict()
        self.logger = logger

    def _round_channels(self, chs):
        if False:
            print('Hello World!')
        return round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    def _make_block(self, ba, block_idx, block_count):
        if False:
            for i in range(10):
                print('nop')
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
        self.in_chs = ba['out_chs']
        return block

    def __call__(self, in_chs, model_block_args):
        if False:
            while True:
                i = 10
        ' Build the blocks\n        Args:\n            in_chs: Number of input-channels passed to first block\n            model_block_args: A list of lists, outer list defines stages, inner\n                list contains strings defining block configuration(s)\n        Return:\n             List of block stacks (each stack wrapped in nn.Sequential)\n        '
        if self.verbose:
            self.logger.info('Building model trunk with %d stages...' % len(model_block_args))
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
            blocks = []
            for (block_idx, block_args) in enumerate(stage_block_args):
                last_block = block_idx == len(stage_block_args) - 1
                extract_features = ''
                if self.verbose:
                    self.logger.info(' Block: {}'.format(block_idx))
                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:
                    block_args['stride'] = 1
                do_extract = False
                if self.feature_location == 'pre_pwl':
                    if last_block:
                        next_stage_idx = stage_idx + 1
                        if next_stage_idx >= len(model_block_args):
                            do_extract = True
                        else:
                            do_extract = model_block_args[next_stage_idx][0]['stride'] > 1
                elif self.feature_location == 'post_exp':
                    if block_args['stride'] > 1 or (last_stack and last_block):
                        do_extract = True
                if do_extract:
                    extract_features = self.feature_location
                next_dilation = current_dilation
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                        if self.verbose:
                            self.logger.info('  Converting stride to dilation to maintain output_stride=={}'.format(self.output_stride))
                    else:
                        current_stride = next_output_stride
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation
                block = self._make_block(block_args, total_block_idx, total_block_count)
                blocks.append(block)
                if extract_features:
                    feature_module = block.feature_module(extract_features)
                    if feature_module:
                        feature_module = 'blocks.{}.{}.'.format(stage_idx, block_idx) + feature_module
                    feature_channels = block.feature_channels(extract_features)
                    self.features[feature_idx] = dict(name=feature_module, num_chs=feature_channels)
                    feature_idx += 1
                total_block_idx += 1
            stages.append(nn.Sequential(*blocks))
        return stages