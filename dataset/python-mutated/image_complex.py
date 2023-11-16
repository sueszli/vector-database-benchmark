from .image import ImageVisual, _APPLY_CLIM_FLOAT, _APPLY_GAMMA_FLOAT
import numpy as np
from .shaders import Function, FunctionChain
COMPLEX_TRANSFORMS = {'real': 'float cplx2float(vec4 data) { return data.r; }', 'imaginary': 'float cplx2float(vec4 data) { return data.g; }', 'magnitude': 'float cplx2float(vec4 data) { return length(vec2(data)); }', 'phase': 'float cplx2float(vec4 data) { return atan(data.g, data.r); }'}
CPU_COMPLEX_TRANSFORMS = {'magnitude': np.abs, 'phase': np.angle, 'real': np.real, 'imaginary': np.imag}

class ComplexImageVisual(ImageVisual):
    """:class:`~vispy.visuals.ImageVisual` subclass displaying a complex-valued image.

    This class handles complex values by using an rg32f float texture behind the scenes,
    storing the real component in the "r"  value and the imaginary in the "g" value.

    Parameters
    ----------
    data : ndarray
        Complex valued ImageVisual data.  Should be a two dimensional array with a dtype
        of np.complex64 or np.complex128.
    complex_mode : str
        The mode used to convert the complex value in each pixel into a scalar:
            * 'real': show only the real component.
            * 'imaginary': show only the imaginary component.
            * 'magnitude': show the magnitude (`np.abs`) of the complex value.
            * 'phase': show the phase (`np.angle`) of the complex value.
    """
    COMPLEX_MODES = set(COMPLEX_TRANSFORMS)

    def __init__(self, data=None, complex_mode='magnitude', **kwargs):
        if False:
            i = 10
            return i + 15
        if complex_mode not in self.COMPLEX_MODES:
            raise ValueError('complex_mode must be one of %s' % ', '.join(self.COMPLEX_MODES))
        self._data_is_complex = np.iscomplexobj(data)
        self._complex_mode = complex_mode
        if kwargs.get('clim', 'auto') == 'auto' and self._data_is_complex:
            kwargs['clim'] = self._calc_complex_clim(data)
        kwargs['texture_format'] = 'r32f' if self._data_is_complex else 'r32f'
        if self._data_is_complex:
            data = self._convert_complex_to_float_view(data)
        super().__init__(data=data, **kwargs)

    def _init_texture(self, data, texture_format, **texture_kwargs):
        if False:
            while True:
                i = 10
        texture_kwargs = {}
        if self._data_is_complex:
            texture_kwargs['format'] = 'rg'
        return super()._init_texture(data, texture_format, **texture_kwargs)

    def set_data(self, image):
        if False:
            print('Hello World!')
        data = np.asarray(image)
        if np.iscomplexobj(data):
            self._data_is_complex = True
            self._texture._format = 'rg'
            data = self._convert_complex_to_float_view(data)
        elif data.ndim == 3 and data.shape[-1] == 2:
            self._data_is_complex = True
        else:
            self._texture._format = None
        return super().set_data(data)

    @staticmethod
    def _convert_complex_to_float_view(complex_arr):
        if False:
            while True:
                i = 10
        complex64_arr = complex_arr.astype(np.complex64, copy=False)
        float_view_arr = complex64_arr.view(dtype=np.float32).reshape(complex64_arr.shape + (2,))
        return float_view_arr

    @property
    def complex_mode(self):
        if False:
            while True:
                i = 10
        return self._data_is_complex and self._complex_mode

    @complex_mode.setter
    def complex_mode(self, value):
        if False:
            for i in range(10):
                print('nop')
        if value not in self.COMPLEX_MODES:
            raise ValueError('complex_mode must be one of %s' % ', '.join(self.COMPLEX_MODES))
        if self._complex_mode != value:
            self._complex_mode = value
            self._need_colortransform_update = True
            self.update()

    def _build_color_transform(self):
        if False:
            while True:
                i = 10
        if self.complex_mode:
            fclim = Function(_APPLY_CLIM_FLOAT)
            fgamma = Function(_APPLY_GAMMA_FLOAT)
            chain = [Function(COMPLEX_TRANSFORMS[self.complex_mode]), fclim, fgamma, Function(self.cmap.glsl_map)]
            fun = FunctionChain(None, chain)
            fclim['clim'] = self._texture.clim_normalized
            fgamma['gamma'] = self.gamma
            return fun
        return super()._build_color_transform()

    @ImageVisual.clim.setter
    def clim(self, clim):
        if False:
            print('Hello World!')
        if clim == 'auto' and self.complex_mode:
            clim = self._calc_complex_clim()
        super(ComplexImageVisual, type(self)).clim.fset(self, clim)

    def _calc_complex_clim(self, data=None):
        if False:
            return 10
        func = CPU_COMPLEX_TRANSFORMS[self.complex_mode]
        _rendered = func(self._data if data is None else data)
        return (_rendered.min(), _rendered.max())