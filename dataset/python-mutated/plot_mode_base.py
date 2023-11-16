import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.color_scheme import ColorScheme
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.utilities.iterables import is_sequence
from time import sleep
from threading import Thread, Event, RLock
import warnings

class PlotModeBase(PlotMode):
    """
    Intended parent class for plotting
    modes. Provides base functionality
    in conjunction with its parent,
    PlotMode.
    """
    '\n    The following attributes are meant\n    to be set at the class level, and serve\n    as parameters to the plot mode registry\n    (in PlotMode). See plot_modes.py for\n    concrete examples.\n    '
    "\n    i_vars\n        'x' for Cartesian2D\n        'xy' for Cartesian3D\n        etc.\n\n    d_vars\n        'y' for Cartesian2D\n        'r' for Polar\n        etc.\n    "
    (i_vars, d_vars) = ('', '')
    '\n    intervals\n        Default intervals for each i_var, and in the\n        same order. Specified [min, max, steps].\n        No variable can be given (it is bound later).\n    '
    intervals = []
    "\n    aliases\n        A list of strings which can be used to\n        access this mode.\n        'cartesian' for Cartesian2D and Cartesian3D\n        'polar' for Polar\n        'cylindrical', 'polar' for Cylindrical\n\n        Note that _init_mode chooses the first alias\n        in the list as the mode's primary_alias, which\n        will be displayed to the end user in certain\n        contexts.\n    "
    aliases = []
    '\n    is_default\n        Whether to set this mode as the default\n        for arguments passed to PlotMode() containing\n        the same number of d_vars as this mode and\n        at most the same number of i_vars.\n    '
    is_default = False
    '\n    All of the above attributes are defined in PlotMode.\n    The following ones are specific to PlotModeBase.\n    '
    '\n    A list of the render styles. Do not modify.\n    '
    styles = {'wireframe': 1, 'solid': 2, 'both': 3}
    '\n    style_override\n        Always use this style if not blank.\n    '
    style_override = ''
    '\n    default_wireframe_color\n    default_solid_color\n        Can be used when color is None or being calculated.\n        Used by PlotCurve and PlotSurface, but not anywhere\n        in PlotModeBase.\n    '
    default_wireframe_color = (0.85, 0.85, 0.85)
    default_solid_color = (0.6, 0.6, 0.9)
    default_rot_preset = 'xy'

    def _get_evaluator(self):
        if False:
            for i in range(10):
                print('nop')
        if self.use_lambda_eval:
            try:
                e = self._get_lambda_evaluator()
                return e
            except Exception:
                warnings.warn('\nWarning: creating lambda evaluator failed. Falling back on SymPy subs evaluator.')
        return self._get_sympy_evaluator()

    def _get_sympy_evaluator(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def _get_lambda_evaluator(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def _on_calculate_verts(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def _on_calculate_cverts(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def __init__(self, *args, bounds_callback=None, **kwargs):
        if False:
            print('Hello World!')
        self.verts = []
        self.cverts = []
        self.bounds = [[S.Infinity, S.NegativeInfinity, 0], [S.Infinity, S.NegativeInfinity, 0], [S.Infinity, S.NegativeInfinity, 0]]
        self.cbounds = [[S.Infinity, S.NegativeInfinity, 0], [S.Infinity, S.NegativeInfinity, 0], [S.Infinity, S.NegativeInfinity, 0]]
        self._draw_lock = RLock()
        self._calculating_verts = Event()
        self._calculating_cverts = Event()
        self._calculating_verts_pos = 0.0
        self._calculating_verts_len = 0.0
        self._calculating_cverts_pos = 0.0
        self._calculating_cverts_len = 0.0
        self._max_render_stack_size = 3
        self._draw_wireframe = [-1]
        self._draw_solid = [-1]
        self._style = None
        self._color = None
        self.predraw = []
        self.postdraw = []
        self.use_lambda_eval = self.options.pop('use_sympy_eval', None) is None
        self.style = self.options.pop('style', '')
        self.color = self.options.pop('color', 'rainbow')
        self.bounds_callback = bounds_callback
        self._on_calculate()

    def synchronized(f):
        if False:
            while True:
                i = 10

        def w(self, *args, **kwargs):
            if False:
                return 10
            self._draw_lock.acquire()
            try:
                r = f(self, *args, **kwargs)
                return r
            finally:
                self._draw_lock.release()
        return w

    @synchronized
    def push_wireframe(self, function):
        if False:
            for i in range(10):
                print('nop')
        '\n        Push a function which performs gl commands\n        used to build a display list. (The list is\n        built outside of the function)\n        '
        assert callable(function)
        self._draw_wireframe.append(function)
        if len(self._draw_wireframe) > self._max_render_stack_size:
            del self._draw_wireframe[1]

    @synchronized
    def push_solid(self, function):
        if False:
            i = 10
            return i + 15
        '\n        Push a function which performs gl commands\n        used to build a display list. (The list is\n        built outside of the function)\n        '
        assert callable(function)
        self._draw_solid.append(function)
        if len(self._draw_solid) > self._max_render_stack_size:
            del self._draw_solid[1]

    def _create_display_list(self, function):
        if False:
            print('Hello World!')
        dl = pgl.glGenLists(1)
        pgl.glNewList(dl, pgl.GL_COMPILE)
        function()
        pgl.glEndList()
        return dl

    def _render_stack_top(self, render_stack):
        if False:
            for i in range(10):
                print('nop')
        top = render_stack[-1]
        if top == -1:
            return -1
        elif callable(top):
            dl = self._create_display_list(top)
            render_stack[-1] = (dl, top)
            return dl
        elif len(top) == 2:
            if pgl.GL_TRUE == pgl.glIsList(top[0]):
                return top[0]
            dl = self._create_display_list(top[1])
            render_stack[-1] = (dl, top[1])
            return dl

    def _draw_solid_display_list(self, dl):
        if False:
            while True:
                i = 10
        pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT)
        pgl.glPolygonMode(pgl.GL_FRONT_AND_BACK, pgl.GL_FILL)
        pgl.glCallList(dl)
        pgl.glPopAttrib()

    def _draw_wireframe_display_list(self, dl):
        if False:
            print('Hello World!')
        pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT)
        pgl.glPolygonMode(pgl.GL_FRONT_AND_BACK, pgl.GL_LINE)
        pgl.glEnable(pgl.GL_POLYGON_OFFSET_LINE)
        pgl.glPolygonOffset(-0.005, -50.0)
        pgl.glCallList(dl)
        pgl.glPopAttrib()

    @synchronized
    def draw(self):
        if False:
            return 10
        for f in self.predraw:
            if callable(f):
                f()
        if self.style_override:
            style = self.styles[self.style_override]
        else:
            style = self.styles[self._style]
        if style & 2:
            dl = self._render_stack_top(self._draw_solid)
            if dl > 0 and pgl.GL_TRUE == pgl.glIsList(dl):
                self._draw_solid_display_list(dl)
        if style & 1:
            dl = self._render_stack_top(self._draw_wireframe)
            if dl > 0 and pgl.GL_TRUE == pgl.glIsList(dl):
                self._draw_wireframe_display_list(dl)
        for f in self.postdraw:
            if callable(f):
                f()

    def _on_change_color(self, color):
        if False:
            print('Hello World!')
        Thread(target=self._calculate_cverts).start()

    def _on_calculate(self):
        if False:
            i = 10
            return i + 15
        Thread(target=self._calculate_all).start()

    def _calculate_all(self):
        if False:
            while True:
                i = 10
        self._calculate_verts()
        self._calculate_cverts()

    def _calculate_verts(self):
        if False:
            while True:
                i = 10
        if self._calculating_verts.is_set():
            return
        self._calculating_verts.set()
        try:
            self._on_calculate_verts()
        finally:
            self._calculating_verts.clear()
        if callable(self.bounds_callback):
            self.bounds_callback()

    def _calculate_cverts(self):
        if False:
            for i in range(10):
                print('nop')
        if self._calculating_verts.is_set():
            return
        while self._calculating_cverts.is_set():
            sleep(0)
        self._calculating_cverts.set()
        try:
            self._on_calculate_cverts()
        finally:
            self._calculating_cverts.clear()

    def _get_calculating_verts(self):
        if False:
            return 10
        return self._calculating_verts.is_set()

    def _get_calculating_verts_pos(self):
        if False:
            i = 10
            return i + 15
        return self._calculating_verts_pos

    def _get_calculating_verts_len(self):
        if False:
            print('Hello World!')
        return self._calculating_verts_len

    def _get_calculating_cverts(self):
        if False:
            while True:
                i = 10
        return self._calculating_cverts.is_set()

    def _get_calculating_cverts_pos(self):
        if False:
            for i in range(10):
                print('nop')
        return self._calculating_cverts_pos

    def _get_calculating_cverts_len(self):
        if False:
            while True:
                i = 10
        return self._calculating_cverts_len

    def _get_style(self):
        if False:
            print('Hello World!')
        return self._style

    @synchronized
    def _set_style(self, v):
        if False:
            print('Hello World!')
        if v is None:
            return
        if v == '':
            step_max = 0
            for i in self.intervals:
                if i.v_steps is None:
                    continue
                step_max = max([step_max, int(i.v_steps)])
            v = ['both', 'solid'][step_max > 40]
        if v not in self.styles:
            raise ValueError('v should be there in self.styles')
        if v == self._style:
            return
        self._style = v

    def _get_color(self):
        if False:
            i = 10
            return i + 15
        return self._color

    @synchronized
    def _set_color(self, v):
        if False:
            for i in range(10):
                print('nop')
        try:
            if v is not None:
                if is_sequence(v):
                    v = ColorScheme(*v)
                else:
                    v = ColorScheme(v)
            if repr(v) == repr(self._color):
                return
            self._on_change_color(v)
            self._color = v
        except Exception as e:
            raise RuntimeError('Color change failed. Reason: %s' % str(e))
    style = property(_get_style, _set_style)
    color = property(_get_color, _set_color)
    calculating_verts = property(_get_calculating_verts)
    calculating_verts_pos = property(_get_calculating_verts_pos)
    calculating_verts_len = property(_get_calculating_verts_len)
    calculating_cverts = property(_get_calculating_cverts)
    calculating_cverts_pos = property(_get_calculating_cverts_pos)
    calculating_cverts_len = property(_get_calculating_cverts_len)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        f = ', '.join((str(d) for d in self.d_vars))
        o = "'mode=%s'" % self.primary_alias
        return ', '.join([f, o])

    def __repr__(self):
        if False:
            return 10
        f = ', '.join((str(d) for d in self.d_vars))
        i = ', '.join((str(i) for i in self.intervals))
        d = [('mode', self.primary_alias), ('color', str(self.color)), ('style', str(self.style))]
        o = "'%s'" % '; '.join(('%s=%s' % (k, v) for (k, v) in d if v != 'None'))
        return ', '.join([f, i, o])