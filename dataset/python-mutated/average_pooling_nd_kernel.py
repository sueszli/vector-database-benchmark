from chainer.functions.pooling import pooling_nd_kernel

class AveragePoolingNDKernelForward(pooling_nd_kernel.PoolingNDKernelForward):

    def name(self):
        if False:
            print('Hello World!')
        return 'avg'

    def in_params(self):
        if False:
            while True:
                i = 10
        return ['T coeff']

    def before(self):
        if False:
            return 10
        return 'T val = 0;'

    def main(self, offset, xs):
        if False:
            i = 10
            return i + 15
        return 'val = val + in[{}];'.format(offset)

    def after(self, out_xs):
        if False:
            print('Hello World!')
        return 'out = val * coeff;'

class AveragePoolingNDKernelBackward(pooling_nd_kernel.PoolingNDKernelBackward):

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'avg'

    def before(self):
        if False:
            print('Hello World!')
        return 'T val = 0;'

    def main(self, offset, xs, out_xs):
        if False:
            print('Hello World!')
        return 'val = val + gy[{}];'.format(offset)

    def after(self, xs):
        if False:
            for i in range(10):
                print('nop')
        return 'gx = val;'