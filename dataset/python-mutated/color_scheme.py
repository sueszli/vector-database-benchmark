from sympy.core.basic import Basic
from sympy.core.symbol import Symbol, symbols
from sympy.utilities.lambdify import lambdify
from .util import interpolate, rinterpolate, create_bounds, update_bounds
from sympy.utilities.iterables import sift

class ColorGradient:
    colors = ([0.4, 0.4, 0.4], [0.9, 0.9, 0.9])
    intervals = (0.0, 1.0)

    def __init__(self, *args):
        if False:
            print('Hello World!')
        if len(args) == 2:
            self.colors = list(args)
            self.intervals = [0.0, 1.0]
        elif len(args) > 0:
            if len(args) % 2 != 0:
                raise ValueError('len(args) should be even')
            self.colors = [args[i] for i in range(1, len(args), 2)]
            self.intervals = [args[i] for i in range(0, len(args), 2)]
        assert len(self.colors) == len(self.intervals)

    def copy(self):
        if False:
            while True:
                i = 10
        c = ColorGradient()
        c.colors = [e[:] for e in self.colors]
        c.intervals = self.intervals[:]
        return c

    def _find_interval(self, v):
        if False:
            i = 10
            return i + 15
        m = len(self.intervals)
        i = 0
        while i < m - 1 and self.intervals[i] <= v:
            i += 1
        return i

    def _interpolate_axis(self, axis, v):
        if False:
            print('Hello World!')
        i = self._find_interval(v)
        v = rinterpolate(self.intervals[i - 1], self.intervals[i], v)
        return interpolate(self.colors[i - 1][axis], self.colors[i][axis], v)

    def __call__(self, r, g, b):
        if False:
            while True:
                i = 10
        c = self._interpolate_axis
        return (c(0, r), c(1, g), c(2, b))
default_color_schemes = {}

class ColorScheme:

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.args = args
        (self.f, self.gradient) = (None, ColorGradient())
        if len(args) == 1 and (not isinstance(args[0], Basic)) and callable(args[0]):
            self.f = args[0]
        elif len(args) == 1 and isinstance(args[0], str):
            if args[0] in default_color_schemes:
                cs = default_color_schemes[args[0]]
                (self.f, self.gradient) = (cs.f, cs.gradient.copy())
            else:
                self.f = lambdify('x,y,z,u,v', args[0])
        else:
            (self.f, self.gradient) = self._interpret_args(args)
        self._test_color_function()
        if not isinstance(self.gradient, ColorGradient):
            raise ValueError('Color gradient not properly initialized. (Not a ColorGradient instance.)')

    def _interpret_args(self, args):
        if False:
            while True:
                i = 10
        (f, gradient) = (None, self.gradient)
        (atoms, lists) = self._sort_args(args)
        s = self._pop_symbol_list(lists)
        s = self._fill_in_vars(s)
        f_str = ', '.join((str(fa) for fa in atoms))
        s_str = (str(sa) for sa in s)
        s_str = ', '.join((sa for sa in s_str if sa.find('unbound') < 0))
        f_error = ValueError('Could not interpret arguments %s as functions of %s.' % (f_str, s_str))
        if len(atoms) == 1:
            fv = atoms[0]
            try:
                f = lambdify(s, [fv, fv, fv])
            except TypeError:
                raise f_error
        elif len(atoms) == 3:
            (fr, fg, fb) = atoms
            try:
                f = lambdify(s, [fr, fg, fb])
            except TypeError:
                raise f_error
        else:
            raise ValueError('A ColorScheme must provide 1 or 3 functions in x, y, z, u, and/or v.')
        if len(lists) == 0:
            gargs = []
        elif len(lists) == 1:
            gargs = lists[0]
        elif len(lists) == 2:
            try:
                ((r1, g1, b1), (r2, g2, b2)) = lists
            except TypeError:
                raise ValueError('If two color arguments are given, they must be given in the format (r1, g1, b1), (r2, g2, b2).')
            gargs = lists
        elif len(lists) == 3:
            try:
                ((r1, r2), (g1, g2), (b1, b2)) = lists
            except Exception:
                raise ValueError('If three color arguments are given, they must be given in the format (r1, r2), (g1, g2), (b1, b2). To create a multi-step gradient, use the syntax [0, colorStart, step1, color1, ..., 1, colorEnd].')
            gargs = [[r1, g1, b1], [r2, g2, b2]]
        else:
            raise ValueError("Don't know what to do with collection arguments %s." % ', '.join((str(l) for l in lists)))
        if gargs:
            try:
                gradient = ColorGradient(*gargs)
            except Exception as ex:
                raise ValueError('Could not initialize a gradient with arguments %s. Inner exception: %s' % (gargs, str(ex)))
        return (f, gradient)

    def _pop_symbol_list(self, lists):
        if False:
            return 10
        symbol_lists = []
        for l in lists:
            mark = True
            for s in l:
                if s is not None and (not isinstance(s, Symbol)):
                    mark = False
                    break
            if mark:
                lists.remove(l)
                symbol_lists.append(l)
        if len(symbol_lists) == 1:
            return symbol_lists[0]
        elif len(symbol_lists) == 0:
            return []
        else:
            raise ValueError('Only one list of Symbols can be given for a color scheme.')

    def _fill_in_vars(self, args):
        if False:
            for i in range(10):
                print('nop')
        defaults = symbols('x,y,z,u,v')
        v_error = ValueError('Could not find what to plot.')
        if len(args) == 0:
            return defaults
        if not isinstance(args, (tuple, list)):
            raise v_error
        if len(args) == 0:
            return defaults
        for s in args:
            if s is not None and (not isinstance(s, Symbol)):
                raise v_error
        vars = [Symbol('unbound%i' % i) for i in range(1, 6)]
        if len(args) == 1:
            vars[3] = args[0]
        elif len(args) == 2:
            if args[0] is not None:
                vars[3] = args[0]
            if args[1] is not None:
                vars[4] = args[1]
        elif len(args) >= 3:
            if args[0] is not None:
                vars[0] = args[0]
            if args[1] is not None:
                vars[1] = args[1]
            if args[2] is not None:
                vars[2] = args[2]
            if len(args) >= 4:
                vars[3] = args[3]
                if len(args) >= 5:
                    vars[4] = args[4]
        return vars

    def _sort_args(self, args):
        if False:
            print('Hello World!')
        (lists, atoms) = sift(args, lambda a: isinstance(a, (tuple, list)), binary=True)
        return (atoms, lists)

    def _test_color_function(self):
        if False:
            return 10
        if not callable(self.f):
            raise ValueError('Color function is not callable.')
        try:
            result = self.f(0, 0, 0, 0, 0)
            if len(result) != 3:
                raise ValueError('length should be equal to 3')
        except TypeError:
            raise ValueError("Color function needs to accept x,y,z,u,v, as arguments even if it doesn't use all of them.")
        except AssertionError:
            raise ValueError('Color function needs to return 3-tuple r,g,b.')
        except Exception:
            pass

    def __call__(self, x, y, z, u, v):
        if False:
            i = 10
            return i + 15
        try:
            return self.f(x, y, z, u, v)
        except Exception:
            return None

    def apply_to_curve(self, verts, u_set, set_len=None, inc_pos=None):
        if False:
            i = 10
            return i + 15
        '\n        Apply this color scheme to a\n        set of vertices over a single\n        independent variable u.\n        '
        bounds = create_bounds()
        cverts = []
        if callable(set_len):
            set_len(len(u_set) * 2)
        for _u in range(len(u_set)):
            if verts[_u] is None:
                cverts.append(None)
            else:
                (x, y, z) = verts[_u]
                (u, v) = (u_set[_u], None)
                c = self(x, y, z, u, v)
                if c is not None:
                    c = list(c)
                    update_bounds(bounds, c)
                cverts.append(c)
            if callable(inc_pos):
                inc_pos()
        for _u in range(len(u_set)):
            if cverts[_u] is not None:
                for _c in range(3):
                    cverts[_u][_c] = rinterpolate(bounds[_c][0], bounds[_c][1], cverts[_u][_c])
                cverts[_u] = self.gradient(*cverts[_u])
            if callable(inc_pos):
                inc_pos()
        return cverts

    def apply_to_surface(self, verts, u_set, v_set, set_len=None, inc_pos=None):
        if False:
            while True:
                i = 10
        '\n        Apply this color scheme to a\n        set of vertices over two\n        independent variables u and v.\n        '
        bounds = create_bounds()
        cverts = []
        if callable(set_len):
            set_len(len(u_set) * len(v_set) * 2)
        for _u in range(len(u_set)):
            column = []
            for _v in range(len(v_set)):
                if verts[_u][_v] is None:
                    column.append(None)
                else:
                    (x, y, z) = verts[_u][_v]
                    (u, v) = (u_set[_u], v_set[_v])
                    c = self(x, y, z, u, v)
                    if c is not None:
                        c = list(c)
                        update_bounds(bounds, c)
                    column.append(c)
                if callable(inc_pos):
                    inc_pos()
            cverts.append(column)
        for _u in range(len(u_set)):
            for _v in range(len(v_set)):
                if cverts[_u][_v] is not None:
                    for _c in range(3):
                        cverts[_u][_v][_c] = rinterpolate(bounds[_c][0], bounds[_c][1], cverts[_u][_v][_c])
                    cverts[_u][_v] = self.gradient(*cverts[_u][_v])
                if callable(inc_pos):
                    inc_pos()
        return cverts

    def str_base(self):
        if False:
            while True:
                i = 10
        return ', '.join((str(a) for a in self.args))

    def __repr__(self):
        if False:
            print('Hello World!')
        return '%s' % self.str_base()
(x, y, z, t, u, v) = symbols('x,y,z,t,u,v')
default_color_schemes['rainbow'] = ColorScheme(z, y, x)
default_color_schemes['zfade'] = ColorScheme(z, (0.4, 0.4, 0.97), (0.97, 0.4, 0.4), (None, None, z))
default_color_schemes['zfade3'] = ColorScheme(z, (None, None, z), [0.0, (0.2, 0.2, 1.0), 0.35, (0.2, 0.8, 0.4), 0.5, (0.3, 0.9, 0.3), 0.65, (0.4, 0.8, 0.2), 1.0, (1.0, 0.2, 0.2)])
default_color_schemes['zfade4'] = ColorScheme(z, (None, None, z), [0.0, (0.3, 0.3, 1.0), 0.3, (0.3, 1.0, 0.3), 0.55, (0.95, 1.0, 0.2), 0.65, (1.0, 0.95, 0.2), 0.85, (1.0, 0.7, 0.2), 1.0, (1.0, 0.3, 0.2)])