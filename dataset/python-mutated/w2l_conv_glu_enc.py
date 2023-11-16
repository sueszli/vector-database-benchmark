import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model, register_model_architecture
from fairseq.modules.fairseq_dropout import FairseqDropout
default_conv_enc_config = '[\n    (400, 13, 170, 0.2),\n    (440, 14, 0, 0.214),\n    (484, 15, 0, 0.22898),\n    (532, 16, 0, 0.2450086),\n    (584, 17, 0, 0.262159202),\n    (642, 18, 0, 0.28051034614),\n    (706, 19, 0, 0.30014607037),\n    (776, 20, 0, 0.321156295296),\n    (852, 21, 0, 0.343637235966),\n    (936, 22, 0, 0.367691842484),\n    (1028, 23, 0, 0.393430271458),\n    (1130, 24, 0, 0.42097039046),\n    (1242, 25, 0, 0.450438317792),\n    (1366, 26, 0, 0.481969000038),\n    (1502, 27, 0, 0.51570683004),\n    (1652, 28, 0, 0.551806308143),\n    (1816, 29, 0, 0.590432749713),\n]'

@register_model('asr_w2l_conv_glu_encoder')
class W2lConvGluEncoderModel(FairseqEncoderModel):

    def __init__(self, encoder):
        if False:
            print('Hello World!')
        super().__init__(encoder)

    @staticmethod
    def add_args(parser):
        if False:
            for i in range(10):
                print('nop')
        'Add model-specific arguments to the parser.'
        parser.add_argument('--input-feat-per-channel', type=int, metavar='N', help='encoder input dimension per input channel')
        parser.add_argument('--in-channels', type=int, metavar='N', help='number of encoder input channels')
        parser.add_argument('--conv-enc-config', type=str, metavar='EXPR', help='\n    an array of tuples each containing the configuration of one conv layer\n    [(out_channels, kernel_size, padding, dropout), ...]\n            ')

    @classmethod
    def build_model(cls, args, task):
        if False:
            return 10
        'Build a new model instance.'
        conv_enc_config = getattr(args, 'conv_enc_config', default_conv_enc_config)
        encoder = W2lConvGluEncoder(vocab_size=len(task.target_dictionary), input_feat_per_channel=args.input_feat_per_channel, in_channels=args.in_channels, conv_enc_config=eval(conv_enc_config))
        return cls(encoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        if False:
            print('Hello World!')
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = False
        return lprobs

class W2lConvGluEncoder(FairseqEncoder):

    def __init__(self, vocab_size, input_feat_per_channel, in_channels, conv_enc_config):
        if False:
            i = 10
            return i + 15
        super().__init__(None)
        self.input_dim = input_feat_per_channel
        if in_channels != 1:
            raise ValueError('only 1 input channel is currently supported')
        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropouts = []
        cur_channels = input_feat_per_channel
        for (out_channels, kernel_size, padding, dropout) in conv_enc_config:
            layer = nn.Conv1d(cur_channels, out_channels, kernel_size, padding=padding)
            layer.weight.data.mul_(math.sqrt(3))
            self.conv_layers.append(nn.utils.weight_norm(layer))
            self.dropouts.append(FairseqDropout(dropout, module_name=self.__class__.__name__))
            if out_channels % 2 != 0:
                raise ValueError('odd # of out_channels is incompatible with GLU')
            cur_channels = out_channels // 2
        for out_channels in [2 * cur_channels, vocab_size]:
            layer = nn.Linear(cur_channels, out_channels)
            layer.weight.data.mul_(math.sqrt(3))
            self.linear_layers.append(nn.utils.weight_norm(layer))
            cur_channels = out_channels // 2

    def forward(self, src_tokens, src_lengths, **kwargs):
        if False:
            return 10
        '\n        src_tokens: padded tensor (B, T, C * feat)\n        src_lengths: tensor of original lengths of input utterances (B,)\n        '
        (B, T, _) = src_tokens.size()
        x = src_tokens.transpose(1, 2).contiguous()
        for layer_idx in range(len(self.conv_layers)):
            x = self.conv_layers[layer_idx](x)
            x = F.glu(x, dim=1)
            x = self.dropouts[layer_idx](x)
        x = x.transpose(1, 2).contiguous()
        x = self.linear_layers[0](x)
        x = F.glu(x, dim=2)
        x = self.dropouts[-1](x)
        x = self.linear_layers[1](x)
        assert x.size(0) == B
        assert x.size(1) == T
        encoder_out = x.transpose(0, 1)
        encoder_padding_mask = (torch.arange(T).view(1, T).expand(B, -1).to(x.device) >= src_lengths.view(B, 1).expand(-1, T)).t()
        return {'encoder_out': encoder_out, 'encoder_padding_mask': encoder_padding_mask}

    def reorder_encoder_out(self, encoder_out, new_order):
        if False:
            print('Hello World!')
        encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        if False:
            while True:
                i = 10
        'Maximum input length supported by the encoder.'
        return (1000000.0, 1000000.0)

@register_model_architecture('asr_w2l_conv_glu_encoder', 'w2l_conv_glu_enc')
def w2l_conv_glu_enc(args):
    if False:
        return 10
    args.input_feat_per_channel = getattr(args, 'input_feat_per_channel', 80)
    args.in_channels = getattr(args, 'in_channels', 1)
    args.conv_enc_config = getattr(args, 'conv_enc_config', default_conv_enc_config)