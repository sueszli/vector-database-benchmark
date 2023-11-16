import six
from chainer.functions.pooling import pooling_nd_kernel
from chainer.utils import conv_nd_kernel

class MaxPoolingNDKernelForward(pooling_nd_kernel.PoolingNDKernelForward):

    def name(self):
        if False:
            while True:
                i = 10
        return 'max'

    def out_params(self):
        if False:
            for i in range(10):
                print('nop')
        return ['S indexes']

    def before(self):
        if False:
            return 10

        def aux(argmax):
            if False:
                while True:
                    i = 10
            return 'int {} = 0;'.format(argmax)
        self.argmaxs = conv_nd_kernel.vars('argmax', self.ndim)
        argmax_decls = conv_nd_kernel.map_(aux, self.argmaxs)
        return '\n'.join(['T maxval = (T)-(1.0/0.0);'] + argmax_decls)

    def main(self, offset, xs):
        if False:
            return 10
        w = conv_nd_kernel.Writer()
        w.write('T v = in[{}];'.format(offset))
        w.write('if (maxval < v) {', 'inc')
        w.write('maxval = v;')
        for (argmax, x) in six.moves.zip(self.argmaxs, xs):
            w.write('{} = {};'.format(argmax, x))
        w.write('}', 'dec')
        return w.get()

    def after(self, out_xs):
        if False:
            print('Hello World!')

        def aux(argmax_k, argmax, p, out_x, s):
            if False:
                return 10
            return 'int {} = {} + {} - {} * {};'.format(argmax_k, argmax, p, out_x, s)
        argmax_ks = conv_nd_kernel.vars('argmax_k', self.ndim)
        argmax_k_decls = conv_nd_kernel.map_(aux, argmax_ks, self.argmaxs, self.ps, out_xs, self.ss)
        indexes_set = 'indexes = {};'.format(conv_nd_kernel.muladdexp(self.ks[1:], argmax_ks[1:], argmax_ks[0]))
        return '\n'.join(['out = maxval;'] + argmax_k_decls + [indexes_set])

class MaxPoolingNDKernelBackward(pooling_nd_kernel.PoolingNDKernelBackward):

    def name(self):
        if False:
            print('Hello World!')
        return 'max'

    def in_params(self):
        if False:
            i = 10
            return i + 15
        return (['raw S indexes'], [])

    def before(self):
        if False:
            return 10
        return 'T val = 0;'

    def main(self, offset, xs, out_xs):
        if False:
            return 10

        def aux(x, out_x, s):
            if False:
                while True:
                    i = 10
            return '{} - {} * {}'.format(x, out_x, s)
        w = conv_nd_kernel.Writer()
        w.write('int kx = {};'.format(conv_nd_kernel.muladdexp(self.ks, conv_nd_kernel.map_(aux, xs, out_xs, self.ss), '0')))
        w.write('if (indexes[{}] == kx) {{'.format(offset), 'inc')
        w.write('val = val + gy[{}];'.format(offset))
        w.write('}', 'dec')
        return w.get()

    def after(self, xs):
        if False:
            print('Hello World!')
        return 'gx = val;'

class MaxPoolingNDKernelForwardWithIndexes(pooling_nd_kernel.PoolingNDKernelForward):

    def name(self):
        if False:
            while True:
                i = 10
        return 'max_index'

    def in_params(self):
        if False:
            return 10
        return ['raw S indexes']

    def out_params(self):
        if False:
            i = 10
            return i + 15
        return []

    def _compile_max_x(self):
        if False:
            for i in range(10):
                print('nop')

        def aux(max_val, out_val, stride_val, pad_val, ksize_vals):
            if False:
                i = 10
                return i + 15
            head = ksize_vals[0]
            tail = ksize_vals[1:]
            if tail:
                command = 'int {} = max(0, {} * {} - {} + index / ({}) % {});'
                return command.format(max_val, out_val, stride_val, pad_val, conv_nd_kernel.mulexp(tail), head)
            else:
                return 'int {} = max(0, {} * {} - {} + index % {});'.format(max_val, out_val, stride_val, pad_val, head)
        max_vals = conv_nd_kernel.vars('max', self.ndim)
        out_vals = conv_nd_kernel.vars('out_x', self.ndim)
        stride_vals = conv_nd_kernel.vars('s', self.ndim)
        pad_vals = conv_nd_kernel.vars('p', self.ndim)
        ksize_vals = conv_nd_kernel.vars('k', self.ndim)
        offset_ks_decls = conv_nd_kernel.map_(aux, max_vals, out_vals, stride_vals, pad_vals, conv_nd_kernel.succ_sublists(ksize_vals))
        return offset_ks_decls

    def _compile_out(self):
        if False:
            print('Hello World!')

        def aux(offset, d_val, max_val, offset1):
            if False:
                for i in range(10):
                    print('nop')
            return 'int {} = {} * ({} + {});'.format(offset, d_val, max_val, offset1)
        d_vals = conv_nd_kernel.vars('d', self.ndim)[1:] + [1]
        max_vals = conv_nd_kernel.vars('max', self.ndim)
        offsets = conv_nd_kernel.vars('offset', self.ndim)
        offsets1 = ['d_0 * c0'] + offsets[:-1]
        offset_strs = conv_nd_kernel.map_(aux, offsets, d_vals, max_vals, offsets1)
        offset_strs.append('out = in[offset_{}];'.format(self.ndim - 1))
        return offset_strs

    def _operation(self):
        if False:
            while True:
                i = 10
        c0 = self._compile_c0()
        (out_x, out_xs) = self._compile_out_x()
        max_x = self._compile_max_x()
        index = ['int index = indexes[i];']
        out = self._compile_out()
        return '\n'.join(c0 + out_x + index + max_x + out)

class MaxPoolingNDKernelForwardWithIndexes1(MaxPoolingNDKernelForward):

    def name(self):
        if False:
            while True:
                i = 10
        return 'max_index1'

    def in_params(self):
        if False:
            return 10
        return ['raw T ggx']

    def out_params(self):
        if False:
            while True:
                i = 10
        return []

    def after(self, out_xs):
        if False:
            i = 10
            return i + 15

        def aux(offset, d_val, max_val, offset1):
            if False:
                i = 10
                return i + 15
            return 'int {} = {} * ({} + {});'.format(offset, d_val, max_val, offset1)
        d_vals = conv_nd_kernel.vars('d', self.ndim)[1:] + [1]
        max_vals = conv_nd_kernel.vars('argmax', self.ndim)
        offsets = conv_nd_kernel.vars('offset', self.ndim)
        offsets1 = ['d_0 * c0'] + offsets[:-1]
        offset_strs = conv_nd_kernel.map_(aux, offsets, d_vals, max_vals, offsets1)
        offset_strs.append('out = ggx[offset_{}];'.format(self.ndim - 1))
        return '\n'.join(offset_strs)