from six import moves
from chainer.backends import cuda
from chainer.utils import conv_nd_kernel

class PoolingNDKernelForward(object):

    def name(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def in_params(self):
        if False:
            return 10
        return []

    def out_params(self):
        if False:
            while True:
                i = 10
        return []

    def before(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def main(self, offset, xs):
        if False:
            return 10
        raise NotImplementedError()

    def after(self, out_xs):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @classmethod
    @cuda.memoize()
    def generate(klass, ndim):
        if False:
            return 10
        return klass()._generate(ndim)

    def _generate(self, ndim):
        if False:
            return 10
        self.ndim = ndim
        self.ds = conv_nd_kernel.vars('d', ndim)
        self.outs = conv_nd_kernel.vars('out', ndim)
        self.ks = conv_nd_kernel.vars('k', ndim)
        self.ss = conv_nd_kernel.vars('s', ndim)
        self.ps = conv_nd_kernel.vars('p', ndim)
        in_params = self._in_params()
        out_params = self._out_params()
        operation = self._operation()
        name = '{}_pool_{}d_fwd'.format(self.name(), self.ndim)
        return (in_params, out_params, operation, name)

    def _in_params(self):
        if False:
            for i in range(10):
                print('nop')

        def aux(x):
            if False:
                return 10
            return 'int32 {}'.format(x)
        in_params = self.in_params()
        if type(in_params) is tuple:
            raws = in_params[0]
            in_params = in_params[1]
        else:
            raws = []
        vars = self.ds + self.outs + self.ks + self.ss + self.ps
        return ', '.join(['raw T in'] + raws + conv_nd_kernel.map_(aux, vars) + in_params)

    def _out_params(self):
        if False:
            print('Hello World!')
        out_params = self.out_params()
        return ', '.join(['T out'] + out_params)

    def _compile_c0(self):
        if False:
            i = 10
            return i + 15
        return ['int c0 = i / ({});'.format(conv_nd_kernel.mulexp(self.outs))]

    def _compile_out_x(self):
        if False:
            i = 10
            return i + 15

        def aux(out_x, outs):
            if False:
                i = 10
                return i + 15
            head = outs[0]
            tail = outs[1:]
            if tail:
                return 'int {} = i / ({}) % {};'.format(out_x, conv_nd_kernel.mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(out_x, head)
        out_xs = conv_nd_kernel.vars('out_x', self.ndim)
        out_xs_decls = conv_nd_kernel.map_(aux, out_xs, conv_nd_kernel.succ_sublists(self.outs))
        return (out_xs_decls, out_xs)

    def _compile_loop(self, out_xs):
        if False:
            print('Hello World!')

        def aux(in_x0, in_x1, d, out, k, s, p):
            if False:
                return 10
            return ['int {} = max(0, {} * {} - {});'.format(in_x0, out, s, p), 'int {} = min({}, {} * {} + {} - {});'.format(in_x1, d, out, s, k, p)]
        in_x0s = conv_nd_kernel.vars('in_x0', self.ndim)
        in_x1s = conv_nd_kernel.vars('in_x1', self.ndim)
        bounds = sum(conv_nd_kernel.map_(aux, in_x0s, in_x1s, self.ds, out_xs, self.ks, self.ss, self.ps), [])

        def _loop_main(main):
            if False:
                while True:
                    i = 10
            w = conv_nd_kernel.Writer()
            xs = conv_nd_kernel.vars('x', self.ndim)
            offsets = conv_nd_kernel.vars('offset', self.ndim)
            ds1 = self.ds[1:] + [1]
            offsets1 = ['d_0 * c0'] + offsets[:-1]
            for (x, in_x0, in_x1, offset, offset1, d1) in moves.zip(xs, in_x0s, in_x1s, offsets, offsets1, ds1):
                w.write('for (int {} = {}; {} < {}; ++{}) {{'.format(x, in_x0, x, in_x1, x), 'inc')
                w.write('int {} = {} * ({} + {});'.format(offset, d1, x, offset1))
            offset = offsets[-1]
            for l in main(offset, xs).split('\n'):
                w.write(l)
            for _ in xs:
                w.write('}', 'dec')
            return [w.get()]
        return (bounds, _loop_main)

    def _compile_procedure(self, out_xs):
        if False:
            for i in range(10):
                print('nop')

        def _main(offset, xs):
            if False:
                i = 10
                return i + 15
            return self.main(offset, xs)
        before = [self.before()]
        after = [self.after(out_xs)]
        return (before, _main, after)

    def _operation(self):
        if False:
            return 10
        c0 = self._compile_c0()
        (out_x, out_xs) = self._compile_out_x()
        (loop_bounds, loop_main) = self._compile_loop(out_xs)
        (before, main, after) = self._compile_procedure(out_xs)
        return '\n'.join(c0 + out_x + loop_bounds + before + loop_main(main) + after)

class PoolingNDKernelBackward(object):

    def name(self):
        if False:
            return 10
        raise NotImplementedError()

    def in_params(self):
        if False:
            while True:
                i = 10
        return []

    def out_params(self):
        if False:
            print('Hello World!')
        return []

    def before(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def main(self, offset, xs, out_xs):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def after(self, xs):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @classmethod
    @cuda.memoize()
    def generate(klass, ndim):
        if False:
            print('Hello World!')
        return klass()._generate(ndim)

    def _generate(self, ndim):
        if False:
            print('Hello World!')
        self.ndim = ndim
        self.ds = conv_nd_kernel.vars('d', ndim)
        self.outs = conv_nd_kernel.vars('out', ndim)
        self.ks = conv_nd_kernel.vars('k', ndim)
        self.ss = conv_nd_kernel.vars('s', ndim)
        self.ps = conv_nd_kernel.vars('p', ndim)
        in_params = self._in_params()
        out_params = self._out_params()
        operation = self._operation()
        name = '{}_pool_{}d_bwd'.format(self.name(), self.ndim)
        return (in_params, out_params, operation, name)

    def _in_params(self):
        if False:
            print('Hello World!')

        def aux(x):
            if False:
                print('Hello World!')
            return 'int32 {}'.format(x)
        in_params = self.in_params()
        if type(in_params) is tuple:
            raws = in_params[0]
            in_params = in_params[1]
        else:
            raws = []
        vars = self.ds + self.outs + self.ks + self.ss + self.ps
        return ', '.join(['raw T gy'] + raws + conv_nd_kernel.map_(aux, vars) + in_params)

    def _out_params(self):
        if False:
            print('Hello World!')
        out_params = self.out_params()
        return ', '.join(['T gx'] + out_params)

    def _compile_c0(self):
        if False:
            return 10
        return ['int c0  = i / ({});'.format(conv_nd_kernel.mulexp(self.ds))]

    def _compile_x(self):
        if False:
            for i in range(10):
                print('nop')

        def aux(x, ds, p):
            if False:
                for i in range(10):
                    print('nop')
            head = ds[0]
            tail = ds[1:]
            if tail:
                return 'int {} = i / ({}) % {} + {};'.format(x, conv_nd_kernel.mulexp(tail), head, p)
            else:
                return 'int {} = i % {} + {};'.format(x, head, p)
        xs = conv_nd_kernel.vars('x', self.ndim)
        xs_decls = conv_nd_kernel.map_(aux, xs, conv_nd_kernel.succ_sublists(self.ds), self.ps)
        return (xs_decls, xs)

    def _compile_loop(self, xs):
        if False:
            print('Hello World!')

        def aux(out_x0, out_x1, x, out, k, s):
            if False:
                return 10
            return ['int {} = max(0, ({} - {} + {}) / {});'.format(out_x0, x, k, s, s), 'int {} = min({}, ({} + {}) / {});'.format(out_x1, out, x, s, s)]
        out_x0s = conv_nd_kernel.vars('out_x0', self.ndim)
        out_x1s = conv_nd_kernel.vars('out_x1', self.ndim)
        bounds = sum(conv_nd_kernel.map_(aux, out_x0s, out_x1s, xs, self.outs, self.ks, self.ss), [])

        def _loop_main(main):
            if False:
                while True:
                    i = 10
            w = conv_nd_kernel.Writer()
            out_xs = conv_nd_kernel.vars('out_x', self.ndim)
            offsets = conv_nd_kernel.vars('offset', self.ndim)
            outs1 = self.outs[1:] + [1]
            offsets1 = ['out_0 * c0'] + offsets[:-1]
            for (out_x, out_x0, out_x1, offset, offset1, out1) in moves.zip(out_xs, out_x0s, out_x1s, offsets, offsets1, outs1):
                w.write('for (int {} = {}; {} < {}; ++{}) {{'.format(out_x, out_x0, out_x, out_x1, out_x), 'inc')
                w.write('int {} = {} * ({} + {});'.format(offset, out1, out_x, offset1))
            offset = offsets[-1]
            for l in main(offset, xs, out_xs).split('\n'):
                w.write(l)
            for _ in out_xs:
                w.write('}', 'dec')
            return [w.get()]
        return (bounds, _loop_main)

    def _compile_procedure(self, xs):
        if False:
            for i in range(10):
                print('nop')

        def _main(offset, xs, out_xs):
            if False:
                while True:
                    i = 10
            return self.main(offset, xs, out_xs)
        before = [self.before()]
        after = [self.after(xs)]
        return (before, _main, after)

    def _operation(self):
        if False:
            print('Hello World!')
        c0 = self._compile_c0()
        (x, xs) = self._compile_x()
        (loop_bounds, loop_main) = self._compile_loop(xs)
        (before, main, after) = self._compile_procedure(xs)
        return '\n'.join(c0 + x + loop_bounds + before + loop_main(main) + after)