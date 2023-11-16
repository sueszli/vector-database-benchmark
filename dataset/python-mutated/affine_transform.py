import numpy as np
import torch as th
import torch.nn as nn
from .layer_base import LayerBase, expect_kaldi_matrix, expect_token_number, to_kaldi_matrix

class AffineTransform(LayerBase):

    def __init__(self, input_dim, output_dim):
        if False:
            print('Hello World!')
        super(AffineTransform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        if False:
            i = 10
            return i + 15
        return self.linear(input)

    def to_kaldi_nnet(self):
        if False:
            for i in range(10):
                print('nop')
        re_str = ''
        re_str += '<AffineTransform> %d %d\n' % (self.output_dim, self.input_dim)
        re_str += '<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0\n'
        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        linear_bias = self.state_dict()['linear.bias']
        x = linear_bias.squeeze().numpy()
        re_str += to_kaldi_matrix(x)
        return re_str

    def to_raw_nnet(self, fid):
        if False:
            for i in range(10):
                print('nop')
        linear_weights = self.state_dict()['linear.weight']
        x = linear_weights.squeeze().numpy()
        x.tofile(fid)
        linear_bias = self.state_dict()['linear.bias']
        x = linear_bias.squeeze().numpy()
        x.tofile(fid)

    def load_kaldi_nnet(self, instr):
        if False:
            while True:
                i = 10
        output = expect_token_number(instr, '<LearnRateCoef>')
        if output is None:
            raise Exception('AffineTransform format error for <LearnRateCoef>')
        (instr, lr) = output
        output = expect_token_number(instr, '<BiasLearnRateCoef>')
        if output is None:
            raise Exception('AffineTransform format error for <BiasLearnRateCoef>')
        (instr, lr) = output
        output = expect_token_number(instr, '<MaxNorm>')
        if output is None:
            raise Exception('AffineTransform format error for <MaxNorm>')
        (instr, lr) = output
        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('AffineTransform format error for parsing matrix')
        (instr, mat) = output
        print(mat.shape)
        self.linear.weight = th.nn.Parameter(th.from_numpy(mat).type(th.FloatTensor))
        output = expect_kaldi_matrix(instr)
        if output is None:
            raise Exception('AffineTransform format error for parsing matrix')
        (instr, mat) = output
        mat = np.squeeze(mat)
        self.linear.bias = th.nn.Parameter(th.from_numpy(mat).type(th.FloatTensor))
        return instr