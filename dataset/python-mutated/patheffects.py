"""
Defines classes for path effects. The path effects are supported in `.Text`,
`.Line2D` and `.Patch`.

.. seealso::
   :ref:`patheffects_guide`
"""
from matplotlib.backend_bases import RendererBase
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms
from matplotlib.path import Path
import numpy as np

class AbstractPathEffect:
    """
    A base class for path effects.

    Subclasses should override the ``draw_path`` method to add effect
    functionality.
    """

    def __init__(self, offset=(0.0, 0.0)):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        offset : (float, float), default: (0, 0)\n            The (x, y) offset to apply to the path, measured in points.\n        '
        self._offset = offset

    def _offset_transform(self, renderer):
        if False:
            i = 10
            return i + 15
        'Apply the offset to the given transform.'
        return mtransforms.Affine2D().translate(*map(renderer.points_to_pixels, self._offset))

    def _update_gc(self, gc, new_gc_dict):
        if False:
            while True:
                i = 10
        '\n        Update the given GraphicsContext with the given dict of properties.\n\n        The keys in the dictionary are used to identify the appropriate\n        ``set_`` method on the *gc*.\n        '
        new_gc_dict = new_gc_dict.copy()
        dashes = new_gc_dict.pop('dashes', None)
        if dashes:
            gc.set_dashes(**dashes)
        for (k, v) in new_gc_dict.items():
            set_method = getattr(gc, 'set_' + k, None)
            if not callable(set_method):
                raise AttributeError(f'Unknown property {k}')
            set_method(v)
        return gc

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        if False:
            return 10
        '\n        Derived should override this method. The arguments are the same\n        as :meth:`matplotlib.backend_bases.RendererBase.draw_path`\n        except the first argument is a renderer.\n        '
        if isinstance(renderer, PathEffectRenderer):
            renderer = renderer._renderer
        return renderer.draw_path(gc, tpath, affine, rgbFace)

class PathEffectRenderer(RendererBase):
    """
    Implements a Renderer which contains another renderer.

    This proxy then intercepts draw calls, calling the appropriate
    :class:`AbstractPathEffect` draw method.

    .. note::
        Not all methods have been overridden on this RendererBase subclass.
        It may be necessary to add further methods to extend the PathEffects
        capabilities further.
    """

    def __init__(self, path_effects, renderer):
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        path_effects : iterable of :class:`AbstractPathEffect`\n            The path effects which this renderer represents.\n        renderer : `~matplotlib.backend_bases.RendererBase` subclass\n\n        '
        self._path_effects = path_effects
        self._renderer = renderer

    def copy_with_path_effect(self, path_effects):
        if False:
            i = 10
            return i + 15
        return self.__class__(path_effects, self._renderer)

    def draw_path(self, gc, tpath, affine, rgbFace=None):
        if False:
            i = 10
            return i + 15
        for path_effect in self._path_effects:
            path_effect.draw_path(self._renderer, gc, tpath, affine, rgbFace)

    def draw_markers(self, gc, marker_path, marker_trans, path, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if len(self._path_effects) == 1:
            return super().draw_markers(gc, marker_path, marker_trans, path, *args, **kwargs)
        for path_effect in self._path_effects:
            renderer = self.copy_with_path_effect([path_effect])
            renderer.draw_markers(gc, marker_path, marker_trans, path, *args, **kwargs)

    def draw_path_collection(self, gc, master_transform, paths, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if len(self._path_effects) == 1:
            return super().draw_path_collection(gc, master_transform, paths, *args, **kwargs)
        for path_effect in self._path_effects:
            renderer = self.copy_with_path_effect([path_effect])
            renderer.draw_path_collection(gc, master_transform, paths, *args, **kwargs)

    def _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath):
        if False:
            for i in range(10):
                print('nop')
        (path, transform) = self._get_text_path_transform(x, y, s, prop, angle, ismath)
        color = gc.get_rgb()
        gc.set_linewidth(0.0)
        self.draw_path(gc, path, transform, rgbFace=color)

    def __getattribute__(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name in ['flipy', 'get_canvas_width_height', 'new_gc', 'points_to_pixels', '_text2path', 'height', 'width']:
            return getattr(self._renderer, name)
        else:
            return object.__getattribute__(self, name)

class Normal(AbstractPathEffect):
    """
    The "identity" PathEffect.

    The Normal PathEffect's sole purpose is to draw the original artist with
    no special path effect.
    """

def _subclass_with_normal(effect_class):
    if False:
        return 10
    '\n    Create a PathEffect class combining *effect_class* and a normal draw.\n    '

    class withEffect(effect_class):

        def draw_path(self, renderer, gc, tpath, affine, rgbFace):
            if False:
                i = 10
                return i + 15
            super().draw_path(renderer, gc, tpath, affine, rgbFace)
            renderer.draw_path(gc, tpath, affine, rgbFace)
    withEffect.__name__ = f'with{effect_class.__name__}'
    withEffect.__qualname__ = f'with{effect_class.__name__}'
    withEffect.__doc__ = f'\n    A shortcut PathEffect for applying `.{effect_class.__name__}` and then\n    drawing the original Artist.\n\n    With this class you can use ::\n\n        artist.set_path_effects([patheffects.with{effect_class.__name__}()])\n\n    as a shortcut for ::\n\n        artist.set_path_effects([patheffects.{effect_class.__name__}(),\n                                 patheffects.Normal()])\n    '
    withEffect.draw_path.__doc__ = effect_class.draw_path.__doc__
    return withEffect

class Stroke(AbstractPathEffect):
    """A line based PathEffect which re-draws a stroke."""

    def __init__(self, offset=(0, 0), **kwargs):
        if False:
            while True:
                i = 10
        '\n        The path will be stroked with its gc updated with the given\n        keyword arguments, i.e., the keyword arguments should be valid\n        gc parameter values.\n        '
        super().__init__(offset)
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        if False:
            return 10
        'Draw the path with updated gc.'
        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)
        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(gc0, tpath, affine + self._offset_transform(renderer), rgbFace)
        gc0.restore()
withStroke = _subclass_with_normal(effect_class=Stroke)

class SimplePatchShadow(AbstractPathEffect):
    """A simple shadow via a filled patch."""

    def __init__(self, offset=(2, -2), shadow_rgbFace=None, alpha=None, rho=0.3, **kwargs):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        offset : (float, float), default: (2, -2)\n            The (x, y) offset of the shadow in points.\n        shadow_rgbFace : color\n            The shadow color.\n        alpha : float, default: 0.3\n            The alpha transparency of the created shadow patch.\n        rho : float, default: 0.3\n            A scale factor to apply to the rgbFace color if *shadow_rgbFace*\n            is not specified.\n        **kwargs\n            Extra keywords are stored and passed through to\n            :meth:`AbstractPathEffect._update_gc`.\n\n        '
        super().__init__(offset)
        if shadow_rgbFace is None:
            self._shadow_rgbFace = shadow_rgbFace
        else:
            self._shadow_rgbFace = mcolors.to_rgba(shadow_rgbFace)
        if alpha is None:
            alpha = 0.3
        self._alpha = alpha
        self._rho = rho
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        if False:
            while True:
                i = 10
        '\n        Overrides the standard draw_path to add the shadow offset and\n        necessary color changes for the shadow.\n        '
        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)
        if self._shadow_rgbFace is None:
            (r, g, b) = (rgbFace or (1.0, 1.0, 1.0))[:3]
            shadow_rgbFace = (r * self._rho, g * self._rho, b * self._rho)
        else:
            shadow_rgbFace = self._shadow_rgbFace
        gc0.set_foreground('none')
        gc0.set_alpha(self._alpha)
        gc0.set_linewidth(0)
        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(gc0, tpath, affine + self._offset_transform(renderer), shadow_rgbFace)
        gc0.restore()
withSimplePatchShadow = _subclass_with_normal(effect_class=SimplePatchShadow)

class SimpleLineShadow(AbstractPathEffect):
    """A simple shadow via a line."""

    def __init__(self, offset=(2, -2), shadow_color='k', alpha=0.3, rho=0.3, **kwargs):
        if False:
            return 10
        "\n        Parameters\n        ----------\n        offset : (float, float), default: (2, -2)\n            The (x, y) offset to apply to the path, in points.\n        shadow_color : color, default: 'black'\n            The shadow color.\n            A value of ``None`` takes the original artist's color\n            with a scale factor of *rho*.\n        alpha : float, default: 0.3\n            The alpha transparency of the created shadow patch.\n        rho : float, default: 0.3\n            A scale factor to apply to the rgbFace color if *shadow_color*\n            is ``None``.\n        **kwargs\n            Extra keywords are stored and passed through to\n            :meth:`AbstractPathEffect._update_gc`.\n        "
        super().__init__(offset)
        if shadow_color is None:
            self._shadow_color = shadow_color
        else:
            self._shadow_color = mcolors.to_rgba(shadow_color)
        self._alpha = alpha
        self._rho = rho
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        if False:
            print('Hello World!')
        '\n        Overrides the standard draw_path to add the shadow offset and\n        necessary color changes for the shadow.\n        '
        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)
        if self._shadow_color is None:
            (r, g, b) = (gc0.get_foreground() or (1.0, 1.0, 1.0))[:3]
            shadow_rgbFace = (r * self._rho, g * self._rho, b * self._rho)
        else:
            shadow_rgbFace = self._shadow_color
        gc0.set_foreground(shadow_rgbFace)
        gc0.set_alpha(self._alpha)
        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(gc0, tpath, affine + self._offset_transform(renderer))
        gc0.restore()

class PathPatchEffect(AbstractPathEffect):
    """
    Draws a `.PathPatch` instance whose Path comes from the original
    PathEffect artist.
    """

    def __init__(self, offset=(0, 0), **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        offset : (float, float), default: (0, 0)\n            The (x, y) offset to apply to the path, in points.\n        **kwargs\n            All keyword arguments are passed through to the\n            :class:`~matplotlib.patches.PathPatch` constructor. The\n            properties which cannot be overridden are "path", "clip_box"\n            "transform" and "clip_path".\n        '
        super().__init__(offset=offset)
        self.patch = mpatches.PathPatch([], **kwargs)

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        if False:
            i = 10
            return i + 15
        self.patch._path = tpath
        self.patch.set_transform(affine + self._offset_transform(renderer))
        self.patch.set_clip_box(gc.get_clip_rectangle())
        clip_path = gc.get_clip_path()
        if clip_path and self.patch.get_clip_path() is None:
            self.patch.set_clip_path(*clip_path)
        self.patch.draw(renderer)

class TickedStroke(AbstractPathEffect):
    """
    A line-based PathEffect which draws a path with a ticked style.

    This line style is frequently used to represent constraints in
    optimization.  The ticks may be used to indicate that one side
    of the line is invalid or to represent a closed boundary of a
    domain (i.e. a wall or the edge of a pipe).

    The spacing, length, and angle of ticks can be controlled.

    This line style is sometimes referred to as a hatched line.

    See also the :doc:`/gallery/misc/tickedstroke_demo` example.
    """

    def __init__(self, offset=(0, 0), spacing=10.0, angle=45.0, length=np.sqrt(2), **kwargs):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        offset : (float, float), default: (0, 0)\n            The (x, y) offset to apply to the path, in points.\n        spacing : float, default: 10.0\n            The spacing between ticks in points.\n        angle : float, default: 45.0\n            The angle between the path and the tick in degrees.  The angle\n            is measured as if you were an ant walking along the curve, with\n            zero degrees pointing directly ahead, 90 to your left, -90\n            to your right, and 180 behind you. To change side of the ticks,\n            change sign of the angle.\n        length : float, default: 1.414\n            The length of the tick relative to spacing.\n            Recommended length = 1.414 (sqrt(2)) when angle=45, length=1.0\n            when angle=90 and length=2.0 when angle=60.\n        **kwargs\n            Extra keywords are stored and passed through to\n            :meth:`AbstractPathEffect._update_gc`.\n\n        Examples\n        --------\n        See :doc:`/gallery/misc/tickedstroke_demo`.\n        '
        super().__init__(offset)
        self._spacing = spacing
        self._angle = angle
        self._length = length
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        if False:
            print('Hello World!')
        'Draw the path with updated gc.'
        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)
        gc0 = self._update_gc(gc0, self._gc)
        trans = affine + self._offset_transform(renderer)
        theta = -np.radians(self._angle)
        trans_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        spacing_px = renderer.points_to_pixels(self._spacing)
        transpath = affine.transform_path(tpath)
        polys = transpath.to_polygons(closed_only=False)
        for p in polys:
            x = p[:, 0]
            y = p[:, 1]
            if x.size < 2:
                continue
            ds = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])
            s = np.concatenate(([0.0], np.cumsum(ds)))
            s_total = s[-1]
            num = int(np.ceil(s_total / spacing_px)) - 1
            s_tick = np.linspace(spacing_px / 2, s_total - spacing_px / 2, num)
            x_tick = np.interp(s_tick, s, x)
            y_tick = np.interp(s_tick, s, y)
            delta_s = self._spacing * 0.001
            u = (np.interp(s_tick + delta_s, s, x) - x_tick) / delta_s
            v = (np.interp(s_tick + delta_s, s, y) - y_tick) / delta_s
            n = np.hypot(u, v)
            mask = n == 0
            n[mask] = 1.0
            uv = np.array([u / n, v / n]).T
            uv[mask] = np.array([0, 0]).T
            dxy = np.dot(uv, trans_matrix) * self._length * spacing_px
            x_end = x_tick + dxy[:, 0]
            y_end = y_tick + dxy[:, 1]
            xyt = np.empty((2 * num, 2), dtype=x_tick.dtype)
            xyt[0::2, 0] = x_tick
            xyt[1::2, 0] = x_end
            xyt[0::2, 1] = y_tick
            xyt[1::2, 1] = y_end
            codes = np.tile([Path.MOVETO, Path.LINETO], num)
            h = Path(xyt, codes)
            renderer.draw_path(gc0, h, affine.inverted() + trans, rgbFace)
        gc0.restore()
withTickedStroke = _subclass_with_normal(effect_class=TickedStroke)