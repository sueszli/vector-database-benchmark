import functools
import six
from chainer.backends import cuda

def mulexp(xs, init=None):
    if False:
        i = 10
        return i + 15
    if init is not None:
        return functools.reduce('{} * {}'.format, xs, init)
    else:
        return functools.reduce('{} * {}'.format, xs)

def andexp(xs, init=None):
    if False:
        return 10
    if init is not None:
        return functools.reduce('{} && {}'.format, xs, init)
    else:
        return functools.reduce('{} && {}'.format, xs)

def muladdexp(xs, ys, init=None):
    if False:
        for i in range(10):
            print('nop')

    def aux(exp, arg):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = arg
        return '({} + {} * {})'.format(y, x, exp)
    if init is not None:
        return functools.reduce(aux, six.moves.zip(xs, ys), init)
    else:
        return functools.reduce(aux, six.moves.zip(xs, ys))

def map_(fn, *lst):
    if False:
        return 10
    return list(map(fn, *lst))

def succ_sublists(xs):
    if False:
        for i in range(10):
            print('nop')
    return [xs[i:] for i in six.moves.range(len(xs))]

def vars(prefix, n):
    if False:
        return 10
    return ['{}_{}'.format(prefix, i) for i in six.moves.range(n)]

class Writer(object):

    def __init__(self):
        if False:
            return 10
        self._indent = 0
        self._lines = []

    def write(self, line, indent=None):
        if False:
            for i in range(10):
                print('nop')
        if indent == 'dec' or indent == 'decinc':
            self._indent -= 1
        self._lines.append('  ' * self._indent + line)
        if indent == 'inc' or indent == 'decinc':
            self._indent += 1

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join(self._lines)

class Im2colNDKernel(object):

    def _in_params(self, ds, outs, ks, ss, ps, dilate):
        if False:
            print('Hello World!')

        def aux(x):
            if False:
                for i in range(10):
                    print('nop')
            return 'int32 {}'.format(x)
        return ', '.join(['raw T img'] + map_(aux, ds + outs + ks + ss + ps + dilate))

    def _out_params(self):
        if False:
            print('Hello World!')
        return 'T col'

    def _compile_c0(self, outs, ks):
        if False:
            print('Hello World!')
        return ['int c0 = i / ({});'.format(mulexp(ks + outs))]

    def _compile_kx(self, ndim, outs, ks):
        if False:
            while True:
                i = 10

        def aux(kx, xs):
            if False:
                return 10
            head = xs[0]
            tail = xs[1:] + outs
            if tail:
                return 'int {} = i / ({}) % {};'.format(kx, mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(kx, head)
        kxs = vars('kx', ndim)
        kx_decls = map_(aux, kxs, succ_sublists(ks))
        return (kx_decls, kxs)

    def _compile_out_x(self, ndim, outs):
        if False:
            i = 10
            return i + 15

        def aux(out_x, xs):
            if False:
                for i in range(10):
                    print('nop')
            head = xs[0]
            tail = xs[1:]
            if tail:
                return 'int {} = i / ({}) % {};'.format(out_x, mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(out_x, head)
        out_xs = vars('out_x', ndim)
        out_x_decls = map_(aux, out_xs, succ_sublists(outs))
        return (out_x_decls, out_xs)

    def _compile_main(self, ndim, ds, ks, ss, ps, dilate, kxs, out_xs):
        if False:
            return 10
        w = Writer()
        ins = vars('in', ndim)
        for (_in, kx, out_x, s, p, di) in six.moves.zip(ins, kxs, out_xs, ss, ps, dilate):
            target = 'int {} = {} * {} + {} * {} - {};'
            w.write(target.format(_in, kx, di, out_x, s, p))

        def rel_aux(_in, d):
            if False:
                return 10
            return '0 <= {} && {} < {}'.format(_in, _in, d)
        w.write('if ({}) {{'.format(andexp(map_(rel_aux, ins, ds))), indent='inc')
        idxs = vars('idx', ndim)
        idx0s = ['c0'] + idxs[:-1]
        for (idx, _in, d, idx0) in six.moves.zip(idxs, ins, ds, idx0s):
            w.write('int {} = {} + {} * {};'.format(idx, _in, d, idx0))
        w.write('col = img[{}];'.format(idxs[-1]))
        w.write('} else {', indent='decinc')
        w.write('col = (T)0;')
        w.write('}', indent='dec')
        return [w.get()]

    def _operation(self, ndim, ds, outs, ks, ss, ps, dilate):
        if False:
            i = 10
            return i + 15
        c0 = self._compile_c0(outs, ks)
        (kx, kxs) = self._compile_kx(ndim, outs, ks)
        (out_x, out_xs) = self._compile_out_x(ndim, outs)
        main = self._compile_main(ndim, ds, ks, ss, ps, dilate, kxs, out_xs)
        return '\n'.join(c0 + kx + out_x + main)

    def _generate(self, ndim):
        if False:
            i = 10
            return i + 15
        ds = vars('d', ndim)
        outs = vars('out', ndim)
        ks = vars('k', ndim)
        ss = vars('s', ndim)
        ps = vars('p', ndim)
        dilate = vars('di', ndim)
        in_params = self._in_params(ds, outs, ks, ss, ps, dilate)
        out_params = self._out_params()
        operation = self._operation(ndim, ds, outs, ks, ss, ps, dilate)
        name = name = 'im2col_{}d'.format(ndim)
        return (in_params, out_params, operation, name)

    @staticmethod
    @cuda.memoize()
    def generate(ndim):
        if False:
            print('Hello World!')
        return _im2col_nd_kernel._generate(ndim)
_im2col_nd_kernel = Im2colNDKernel()

class Col2imNDKernel(object):

    def _in_params(self, ds, outs, ks, ss, ps, dilate):
        if False:
            i = 10
            return i + 15

        def aux(x):
            if False:
                return 10
            return 'int32 {}'.format(x)
        return ', '.join(['raw T col'] + map_(aux, ds + outs + ks + ss + ps + dilate))

    def _out_params(self):
        if False:
            return 10
        return 'T img'

    def _compile_c0(self, ds):
        if False:
            print('Hello World!')
        return ['int c0 = i / ({});'.format(mulexp(ds))]

    def _compile_x(self, ndim, ds):
        if False:
            for i in range(10):
                print('nop')

        def aux(x, ds):
            if False:
                print('Hello World!')
            head = ds[0]
            tail = ds[1:]
            if tail:
                return 'int {} = i / ({}) % {};'.format(x, mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(x, head)
        xs = vars('x', ndim)
        x_decls = map_(aux, xs, succ_sublists(ds))
        return (x_decls, xs)

    def _compile_loop(self, ndim, outs, ks, ss, ps, xs, dilate):
        if False:
            for i in range(10):
                print('nop')

        def _loop_main(main, ndim, ks, ss):
            if False:
                i = 10
                return i + 15
            w = Writer()
            out_xs = vars('out_x', ndim)
            kxs = vars('kx', ndim)
            for (out, out_x, kx, s, p, x, k, di) in six.moves.zip(outs, out_xs, kxs, ss, ps, xs, ks, dilate):
                w.write('for (int {} = 0; {} < {}; ++{}) {{'.format(kx, kx, k, kx), indent='inc')
                w.write('int {} = {} + {} - {} * {};'.format(out_x, x, p, kx, di))
                w.write('if (0 > {} || {} >= {} * {}) continue;'.format(out_x, out_x, out, s))
                w.write('if ({} % {} != 0) continue;'.format(out_x, s))
                w.write('{} /= {};'.format(out_x, s))
            for l in main(ks, kxs, out_xs).split('\n'):
                w.write(l)
            for _ in out_xs:
                w.write('}', indent='dec')
            return [w.get()]
        return _loop_main

    def _compile_procedure(self, outs, xs):
        if False:
            while True:
                i = 10

        def _main(ks, kxs, out_xs):
            if False:
                print('Hello World!')
            index = muladdexp(outs, out_xs, muladdexp(ks, kxs, 'c0'))
            return 'val = val + col[{}];'.format(index)
        before = ['T val = 0;']
        after = ['img = val;']
        return (before, _main, after)

    def _operation(self, ndim, ds, outs, ks, ss, ps, dilate):
        if False:
            i = 10
            return i + 15
        c0 = self._compile_c0(ds)
        (x, xs) = self._compile_x(ndim, ds)
        loop_main = self._compile_loop(ndim, outs, ks, ss, ps, xs, dilate)
        (before, main, after) = self._compile_procedure(outs, xs)
        return '\n'.join(c0 + x + before + loop_main(main, ndim, ks, ss) + after)

    def _generate(self, ndim):
        if False:
            print('Hello World!')
        ds = vars('d', ndim)
        outs = vars('out', ndim)
        ks = vars('k', ndim)
        ss = vars('s', ndim)
        ps = vars('p', ndim)
        dilate = vars('di', ndim)
        in_params = self._in_params(ds, outs, ks, ss, ps, dilate)
        out_params = self._out_params()
        operation = self._operation(ndim, ds, outs, ks, ss, ps, dilate)
        name = 'col2im_{}d'.format(ndim)
        return (in_params, out_params, operation, name)

    @staticmethod
    @cuda.memoize()
    def generate(ndim):
        if False:
            while True:
                i = 10
        return _col2im_nd_kernel._generate(ndim)
_col2im_nd_kernel = Col2imNDKernel()