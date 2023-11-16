import argparse
import math
import megengine.functional as F
import megengine.module as M
import numpy as np
from megengine import jit, tensor

class ConvNet(M.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.conv1 = M.Conv2d(in_channels=3, out_channels=1, kernel_size=3, bias=False)

    def forward(self, input):
        if False:
            print('Hello World!')
        x = self.conv1(input)
        return x
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dump mge model for add_demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', help='set the dir where the model to dump', default='.', type=str)
    args = parser.parse_args()
    net = ConvNet()
    net.eval()

    @jit.trace(symbolic=True, capture_as_const=True)
    def fun(data):
        if False:
            print('Hello World!')
        return net(data)
    inp = tensor(np.arange(0, 96).astype('float32').reshape(2, 3, 4, 4))
    out = fun(inp)
    fun.dump(args.dir + '/conv_demo_f32_without_data.mge', arg_names=['data'], no_assert=True)