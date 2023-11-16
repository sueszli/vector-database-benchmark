from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence

class PlotMode(PlotObject):
    """
    Grandparent class for plotting
    modes. Serves as interface for
    registration, lookup, and init
    of modes.

    To create a new plot mode,
    inherit from PlotModeBase
    or one of its children, such
    as PlotSurface or PlotCurve.
    """
    (i_vars, d_vars) = ('', '')
    intervals = []
    aliases = []
    is_default = False

    def draw(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError()
    _mode_alias_list = []
    _mode_map = {1: {1: {}, 2: {}}, 2: {1: {}, 2: {}}, 3: {1: {}, 2: {}}}
    _mode_default_map = {1: {}, 2: {}, 3: {}}
    (_i_var_max, _d_var_max) = (2, 3)

    def __new__(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        This is the function which interprets\n        arguments given to Plot.__init__ and\n        Plot.__setattr__. Returns an initialized\n        instance of the appropriate child class.\n        '
        (newargs, newkwargs) = PlotMode._extract_options(args, kwargs)
        mode_arg = newkwargs.get('mode', '')
        (d_vars, intervals) = PlotMode._interpret_args(newargs)
        i_vars = PlotMode._find_i_vars(d_vars, intervals)
        (i, d) = (max([len(i_vars), len(intervals)]), len(d_vars))
        subcls = PlotMode._get_mode(mode_arg, i, d)
        o = object.__new__(subcls)
        o.d_vars = d_vars
        o._fill_i_vars(i_vars)
        o._fill_intervals(intervals)
        o.options = newkwargs
        return o

    @staticmethod
    def _get_mode(mode_arg, i_var_count, d_var_count):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tries to return an appropriate mode class.\n        Intended to be called only by __new__.\n\n        mode_arg\n            Can be a string or a class. If it is a\n            PlotMode subclass, it is simply returned.\n            If it is a string, it can an alias for\n            a mode or an empty string. In the latter\n            case, we try to find a default mode for\n            the i_var_count and d_var_count.\n\n        i_var_count\n            The number of independent variables\n            needed to evaluate the d_vars.\n\n        d_var_count\n            The number of dependent variables;\n            usually the number of functions to\n            be evaluated in plotting.\n\n        For example, a Cartesian function y = f(x) has\n        one i_var (x) and one d_var (y). A parametric\n        form x,y,z = f(u,v), f(u,v), f(u,v) has two\n        two i_vars (u,v) and three d_vars (x,y,z).\n        '
        try:
            m = None
            if issubclass(mode_arg, PlotMode):
                m = mode_arg
        except TypeError:
            pass
        if m:
            if not m._was_initialized:
                raise ValueError('To use unregistered plot mode %s you must first call %s._init_mode().' % (m.__name__, m.__name__))
            if d_var_count != m.d_var_count:
                raise ValueError('%s can only plot functions with %i dependent variables.' % (m.__name__, m.d_var_count))
            if i_var_count > m.i_var_count:
                raise ValueError('%s cannot plot functions with more than %i independent variables.' % (m.__name__, m.i_var_count))
            return m
        if isinstance(mode_arg, str):
            (i, d) = (i_var_count, d_var_count)
            if i > PlotMode._i_var_max:
                raise ValueError(var_count_error(True, True))
            if d > PlotMode._d_var_max:
                raise ValueError(var_count_error(False, True))
            if not mode_arg:
                return PlotMode._get_default_mode(i, d)
            else:
                return PlotMode._get_aliased_mode(mode_arg, i, d)
        else:
            raise ValueError('PlotMode argument must be a class or a string')

    @staticmethod
    def _get_default_mode(i, d, i_vars=-1):
        if False:
            while True:
                i = 10
        if i_vars == -1:
            i_vars = i
        try:
            return PlotMode._mode_default_map[d][i]
        except KeyError:
            if i < PlotMode._i_var_max:
                return PlotMode._get_default_mode(i + 1, d, i_vars)
            else:
                raise ValueError("Couldn't find a default mode for %i independent and %i dependent variables." % (i_vars, d))

    @staticmethod
    def _get_aliased_mode(alias, i, d, i_vars=-1):
        if False:
            i = 10
            return i + 15
        if i_vars == -1:
            i_vars = i
        if alias not in PlotMode._mode_alias_list:
            raise ValueError("Couldn't find a mode called %s. Known modes: %s." % (alias, ', '.join(PlotMode._mode_alias_list)))
        try:
            return PlotMode._mode_map[d][i][alias]
        except TypeError:
            if i < PlotMode._i_var_max:
                return PlotMode._get_aliased_mode(alias, i + 1, d, i_vars)
            else:
                raise ValueError("Couldn't find a %s mode for %i independent and %i dependent variables." % (alias, i_vars, d))

    @classmethod
    def _register(cls):
        if False:
            return 10
        '\n        Called once for each user-usable plot mode.\n        For Cartesian2D, it is invoked after the\n        class definition: Cartesian2D._register()\n        '
        name = cls.__name__
        cls._init_mode()
        try:
            (i, d) = (cls.i_var_count, cls.d_var_count)
            for a in cls.aliases:
                if a not in PlotMode._mode_alias_list:
                    PlotMode._mode_alias_list.append(a)
                PlotMode._mode_map[d][i][a] = cls
            if cls.is_default:
                PlotMode._mode_default_map[d][i] = cls
        except Exception as e:
            raise RuntimeError('Failed to register plot mode %s. Reason: %s' % (name, str(e)))

    @classmethod
    def _init_mode(cls):
        if False:
            i = 10
            return i + 15
        "\n        Initializes the plot mode based on\n        the 'mode-specific parameters' above.\n        Only intended to be called by\n        PlotMode._register(). To use a mode without\n        registering it, you can directly call\n        ModeSubclass._init_mode().\n        "

        def symbols_list(symbol_str):
            if False:
                i = 10
                return i + 15
            return [Symbol(s) for s in symbol_str]
        cls.i_vars = symbols_list(cls.i_vars)
        cls.d_vars = symbols_list(cls.d_vars)
        cls.i_var_count = len(cls.i_vars)
        cls.d_var_count = len(cls.d_vars)
        if cls.i_var_count > PlotMode._i_var_max:
            raise ValueError(var_count_error(True, False))
        if cls.d_var_count > PlotMode._d_var_max:
            raise ValueError(var_count_error(False, False))
        if len(cls.aliases) > 0:
            cls.primary_alias = cls.aliases[0]
        else:
            cls.primary_alias = cls.__name__
        di = cls.intervals
        if len(di) != cls.i_var_count:
            raise ValueError('Plot mode must provide a default interval for each i_var.')
        for i in range(cls.i_var_count):
            if len(di[i]) != 3:
                raise ValueError('length should be equal to 3')
            di[i] = PlotInterval(None, *di[i])
        cls._was_initialized = True
    _was_initialized = False

    @staticmethod
    def _find_i_vars(functions, intervals):
        if False:
            print('Hello World!')
        i_vars = []
        for i in intervals:
            if i.v is None:
                continue
            elif i.v in i_vars:
                raise ValueError('Multiple intervals given for %s.' % str(i.v))
            i_vars.append(i.v)
        for f in functions:
            for a in f.free_symbols:
                if a not in i_vars:
                    i_vars.append(a)
        return i_vars

    def _fill_i_vars(self, i_vars):
        if False:
            while True:
                i = 10
        self.i_vars = [Symbol(str(i)) for i in self.i_vars]
        for i in range(len(i_vars)):
            self.i_vars[i] = i_vars[i]

    def _fill_intervals(self, intervals):
        if False:
            return 10
        self.intervals = [PlotInterval(i) for i in self.intervals]
        v_used = []
        for i in range(len(intervals)):
            self.intervals[i].fill_from(intervals[i])
            if self.intervals[i].v is not None:
                v_used.append(self.intervals[i].v)
        for i in range(len(self.intervals)):
            if self.intervals[i].v is None:
                u = [v for v in self.i_vars if v not in v_used]
                if len(u) == 0:
                    raise ValueError('length should not be equal to 0')
                self.intervals[i].v = u[0]
                v_used.append(u[0])

    @staticmethod
    def _interpret_args(args):
        if False:
            return 10
        interval_wrong_order = 'PlotInterval %s was given before any function(s).'
        interpret_error = 'Could not interpret %s as a function or interval.'
        (functions, intervals) = ([], [])
        if isinstance(args[0], GeometryEntity):
            for coords in list(args[0].arbitrary_point()):
                functions.append(coords)
            intervals.append(PlotInterval.try_parse(args[0].plot_interval()))
        else:
            for a in args:
                i = PlotInterval.try_parse(a)
                if i is not None:
                    if len(functions) == 0:
                        raise ValueError(interval_wrong_order % str(i))
                    else:
                        intervals.append(i)
                else:
                    if is_sequence(a, include=str):
                        raise ValueError(interpret_error % str(a))
                    try:
                        f = sympify(a)
                        functions.append(f)
                    except TypeError:
                        raise ValueError(interpret_error % str(a))
        return (functions, intervals)

    @staticmethod
    def _extract_options(args, kwargs):
        if False:
            print('Hello World!')
        (newkwargs, newargs) = ({}, [])
        for a in args:
            if isinstance(a, str):
                newkwargs = dict(newkwargs, **parse_option_string(a))
            else:
                newargs.append(a)
        newkwargs = dict(newkwargs, **kwargs)
        return (newargs, newkwargs)

def var_count_error(is_independent, is_plotting):
    if False:
        while True:
            i = 10
    '\n    Used to format an error message which differs\n    slightly in 4 places.\n    '
    if is_plotting:
        v = 'Plotting'
    else:
        v = 'Registering plot modes'
    if is_independent:
        (n, s) = (PlotMode._i_var_max, 'independent')
    else:
        (n, s) = (PlotMode._d_var_max, 'dependent')
    return '%s with more than %i %s variables is not supported.' % (v, n, s)