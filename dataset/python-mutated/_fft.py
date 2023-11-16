from numpy import prod
import cupy
from cupy.fft import config
from cupy.fft._fft import _convert_fft_type, _default_fft_func, _fft, _get_cufft_plan_nd, _get_fftn_out_size, _output_dtype
from cupy.fft._cache import get_plan_cache

def get_fft_plan(a, shape=None, axes=None, value_type='C2C'):
    if False:
        print('Hello World!')
    " Generate a CUDA FFT plan for transforming up to three axes.\n\n    Args:\n        a (cupy.ndarray): Array to be transform, assumed to be either C- or\n            F- contiguous.\n        shape (None or tuple of ints): Shape of the transformed axes of the\n            output. If ``shape`` is not given, the lengths of the input along\n            the axes specified by ``axes`` are used.\n        axes (None or int or tuple of int):  The axes of the array to\n            transform. If `None`, it is assumed that all axes are transformed.\n\n            Currently, for performing N-D transform these must be a set of up\n            to three adjacent axes, and must include either the first or the\n            last axis of the array.\n        value_type (str): The FFT type to perform. Acceptable values are:\n\n            * 'C2C': complex-to-complex transform (default)\n            * 'R2C': real-to-complex transform\n            * 'C2R': complex-to-real transform\n\n    Returns:\n        a cuFFT plan for either 1D transform (``cupy.cuda.cufft.Plan1d``) or\n        N-D transform (``cupy.cuda.cufft.PlanNd``).\n\n    .. note::\n        The returned plan can not only be passed as one of the arguments of\n        the functions in ``cupyx.scipy.fftpack``, but also be used as a\n        context manager for both ``cupy.fft`` and ``cupyx.scipy.fftpack``\n        functions:\n\n        .. code-block:: python\n\n            x = cupy.random.random(16).reshape(4, 4).astype(complex)\n            plan = cupyx.scipy.fftpack.get_fft_plan(x)\n            with plan:\n                y = cupy.fft.fftn(x)\n                # alternatively:\n                y = cupyx.scipy.fftpack.fftn(x)  # no explicit plan is given!\n            # alternatively:\n            y = cupyx.scipy.fftpack.fftn(x, plan=plan)  # pass plan explicitly\n\n        In the first case, no cuFFT plan will be generated automatically,\n        even if ``cupy.fft.config.enable_nd_planning = True`` is set.\n\n    .. note::\n        If this function is called under the context of\n        :func:`~cupy.fft.config.set_cufft_callbacks`, the generated plan will\n        have callbacks enabled.\n\n    .. warning::\n        This API is a deviation from SciPy's, is currently experimental, and\n        may be changed in the future version.\n    "
    from cupy.cuda import cufft
    if a.flags.c_contiguous:
        order = 'C'
    elif a.flags.f_contiguous:
        order = 'F'
    else:
        raise ValueError('Input array a must be contiguous')
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(axes, int):
        axes = (axes,)
    if shape is not None and axes is not None and (len(shape) != len(axes)):
        raise ValueError('Shape and axes have different lengths.')
    if axes is None:
        n = a.ndim if shape is None else len(shape)
        axes = tuple((i for i in range(-n, 0)))
        if n == 1:
            axis1D = 0
    else:
        n = len(axes)
        if n == 1:
            axis1D = axes[0]
            if axis1D >= a.ndim or axis1D < -a.ndim:
                err = 'The chosen axis ({0}) exceeds the number of dimensions of a ({1})'.format(axis1D, a.ndim)
                raise ValueError(err)
        elif n > 3:
            raise ValueError('Only up to three axes is supported')
    transformed_shape = shape
    shape = list(a.shape)
    if transformed_shape is not None:
        for (s, axis) in zip(transformed_shape, axes):
            if s is not None:
                if axis == axes[-1] and value_type == 'C2R':
                    s = s // 2 + 1
                shape[axis] = s
    shape = tuple(shape)
    out_dtype = _output_dtype(a.dtype, value_type)
    fft_type = _convert_fft_type(out_dtype, value_type)
    if n > 1 and value_type != 'C2C' and a.flags.f_contiguous:
        raise ValueError('C2R/R2C PlanNd for F-order arrays is not supported')
    if n > 1:
        if cupy.cuda.runtime.is_hip and value_type == 'C2R':
            raise RuntimeError("hipFFT's C2R PlanNd is buggy and unsupported")
        out_size = _get_fftn_out_size(shape, transformed_shape, axes[-1], value_type)
        plan = _get_cufft_plan_nd(shape, fft_type, axes=axes, order=order, out_size=out_size, to_cache=False)
    else:
        if value_type != 'C2R':
            out_size = shape[axis1D]
        else:
            out_size = _get_fftn_out_size(shape, transformed_shape, axis1D, value_type)
        batch = prod(shape) // shape[axis1D]
        devices = None if not config.use_multi_gpus else config._devices
        keys = (out_size, fft_type, batch, devices)
        mgr = config.get_current_callback_manager()
        if mgr is not None:
            load_aux = mgr.cb_load_aux_arr
            store_aux = mgr.cb_store_aux_arr
            keys += (mgr.cb_load, mgr.cb_store, 0 if load_aux is None else load_aux.data.ptr, 0 if store_aux is None else store_aux.data.ptr)
        cache = get_plan_cache()
        cached_plan = cache.get(keys)
        if cached_plan is not None:
            plan = cached_plan
        elif mgr is None:
            plan = cufft.Plan1d(out_size, fft_type, batch, devices=devices)
        else:
            if devices:
                raise NotImplementedError('multi-GPU cuFFT callbacks are not yet supported')
            plan = mgr.create_plan(('Plan1d', keys[:-3]))
            mgr.set_callbacks(plan)
    return plan

def fft(x, n=None, axis=-1, overwrite_x=False, plan=None):
    if False:
        i = 10
        return i + 15
    'Compute the one-dimensional FFT.\n\n    Args:\n        x (cupy.ndarray): Array to be transformed.\n        n (None or int): Length of the transformed axis of the output. If ``n``\n            is not given, the length of the input along the axis specified by\n            ``axis`` is used.\n        axis (int): Axis over which to compute the FFT.\n        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.\n        plan (:class:`cupy.cuda.cufft.Plan1d` or ``None``): a cuFFT plan for\n            transforming ``x`` over ``axis``, which can be obtained using::\n\n                plan = cupyx.scipy.fftpack.get_fft_plan(x, axis)\n\n            Note that `plan` is defaulted to None, meaning CuPy will use an\n            auto-generated plan behind the scene.\n\n    Returns:\n        cupy.ndarray:\n            The transformed array which shape is specified by ``n`` and type\n            will convert to complex if that of the input is another.\n\n    .. note::\n       The argument `plan` is currently experimental and the interface may be\n       changed in the future version.\n\n    .. seealso:: :func:`scipy.fftpack.fft`\n    '
    from cupy.cuda import cufft
    return _fft(x, (n,), (axis,), None, cufft.CUFFT_FORWARD, overwrite_x=overwrite_x, plan=plan)

def ifft(x, n=None, axis=-1, overwrite_x=False, plan=None):
    if False:
        for i in range(10):
            print('nop')
    'Compute the one-dimensional inverse FFT.\n\n    Args:\n        x (cupy.ndarray): Array to be transformed.\n        n (None or int): Length of the transformed axis of the output. If ``n``\n            is not given, the length of the input along the axis specified by\n            ``axis`` is used.\n        axis (int): Axis over which to compute the FFT.\n        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.\n        plan (:class:`cupy.cuda.cufft.Plan1d` or ``None``): a cuFFT plan for\n            transforming ``x`` over ``axis``, which can be obtained using::\n\n                plan = cupyx.scipy.fftpack.get_fft_plan(x, axis)\n\n            Note that `plan` is defaulted to None, meaning CuPy will use an\n            auto-generated plan behind the scene.\n\n    Returns:\n        cupy.ndarray:\n            The transformed array which shape is specified by ``n`` and type\n            will convert to complex if that of the input is another.\n\n    .. note::\n       The argument `plan` is currently experimental and the interface may be\n       changed in the future version.\n\n    .. seealso:: :func:`scipy.fftpack.ifft`\n    '
    from cupy.cuda import cufft
    return _fft(x, (n,), (axis,), None, cufft.CUFFT_INVERSE, overwrite_x=overwrite_x, plan=plan)

def fft2(x, shape=None, axes=(-2, -1), overwrite_x=False, plan=None):
    if False:
        print('Hello World!')
    'Compute the two-dimensional FFT.\n\n    Args:\n        x (cupy.ndarray): Array to be transformed.\n        shape (None or tuple of ints): Shape of the transformed axes of the\n            output. If ``shape`` is not given, the lengths of the input along\n            the axes specified by ``axes`` are used.\n        axes (tuple of ints): Axes over which to compute the FFT.\n        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.\n        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for\n            transforming ``x`` over ``axes``, which can be obtained using::\n\n                plan = cupyx.scipy.fftpack.get_fft_plan(x, axes)\n\n            Note that `plan` is defaulted to None, meaning CuPy will either\n            use an auto-generated plan behind the scene if cupy.fft.config.\n            enable_nd_planning = True, or use no cuFFT plan if it is set to\n            False.\n\n    Returns:\n        cupy.ndarray:\n            The transformed array which shape is specified by ``shape`` and\n            type will convert to complex if that of the input is another.\n\n    .. seealso:: :func:`scipy.fftpack.fft2`\n\n    .. note::\n       The argument `plan` is currently experimental and the interface may be\n       changed in the future version.\n    '
    from cupy.cuda import cufft
    func = _default_fft_func(x, shape, axes, plan)
    return func(x, shape, axes, None, cufft.CUFFT_FORWARD, overwrite_x=overwrite_x, plan=plan)

def ifft2(x, shape=None, axes=(-2, -1), overwrite_x=False, plan=None):
    if False:
        for i in range(10):
            print('nop')
    'Compute the two-dimensional inverse FFT.\n\n    Args:\n        x (cupy.ndarray): Array to be transformed.\n        shape (None or tuple of ints): Shape of the transformed axes of the\n            output. If ``shape`` is not given, the lengths of the input along\n            the axes specified by ``axes`` are used.\n        axes (tuple of ints): Axes over which to compute the FFT.\n        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.\n        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for\n            transforming ``x`` over ``axes``, which can be obtained using::\n\n                plan = cupyx.scipy.fftpack.get_fft_plan(x, axes)\n\n            Note that `plan` is defaulted to None, meaning CuPy will either\n            use an auto-generated plan behind the scene if cupy.fft.config.\n            enable_nd_planning = True, or use no cuFFT plan if it is set to\n            False.\n\n    Returns:\n        cupy.ndarray:\n            The transformed array which shape is specified by ``shape`` and\n            type will convert to complex if that of the input is another.\n\n    .. seealso:: :func:`scipy.fftpack.ifft2`\n\n    .. note::\n       The argument `plan` is currently experimental and the interface may be\n       changed in the future version.\n    '
    from cupy.cuda import cufft
    func = _default_fft_func(x, shape, axes, plan)
    return func(x, shape, axes, None, cufft.CUFFT_INVERSE, overwrite_x=overwrite_x, plan=plan)

def fftn(x, shape=None, axes=None, overwrite_x=False, plan=None):
    if False:
        i = 10
        return i + 15
    'Compute the N-dimensional FFT.\n\n    Args:\n        x (cupy.ndarray): Array to be transformed.\n        shape (None or tuple of ints): Shape of the transformed axes of the\n            output. If ``shape`` is not given, the lengths of the input along\n            the axes specified by ``axes`` are used.\n        axes (tuple of ints): Axes over which to compute the FFT.\n        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.\n        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for\n            transforming ``x`` over ``axes``, which can be obtained using::\n\n                plan = cupyx.scipy.fftpack.get_fft_plan(x, axes)\n\n            Note that `plan` is defaulted to None, meaning CuPy will either\n            use an auto-generated plan behind the scene if cupy.fft.config.\n            enable_nd_planning = True, or use no cuFFT plan if it is set to\n            False.\n\n    Returns:\n        cupy.ndarray:\n            The transformed array which shape is specified by ``shape`` and\n            type will convert to complex if that of the input is another.\n\n    .. seealso:: :func:`scipy.fftpack.fftn`\n\n    .. note::\n       The argument `plan` is currently experimental and the interface may be\n       changed in the future version.\n    '
    from cupy.cuda import cufft
    func = _default_fft_func(x, shape, axes, plan)
    return func(x, shape, axes, None, cufft.CUFFT_FORWARD, overwrite_x=overwrite_x, plan=plan)

def ifftn(x, shape=None, axes=None, overwrite_x=False, plan=None):
    if False:
        print('Hello World!')
    'Compute the N-dimensional inverse FFT.\n\n    Args:\n        x (cupy.ndarray): Array to be transformed.\n        shape (None or tuple of ints): Shape of the transformed axes of the\n            output. If ``shape`` is not given, the lengths of the input along\n            the axes specified by ``axes`` are used.\n        axes (tuple of ints): Axes over which to compute the FFT.\n        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.\n        plan (:class:`cupy.cuda.cufft.PlanNd` or ``None``): a cuFFT plan for\n            transforming ``x`` over ``axes``, which can be obtained using::\n\n                plan = cupyx.scipy.fftpack.get_fft_plan(x, axes)\n\n            Note that `plan` is defaulted to None, meaning CuPy will either\n            use an auto-generated plan behind the scene if cupy.fft.config.\n            enable_nd_planning = True, or use no cuFFT plan if it is set to\n            False.\n\n    Returns:\n        cupy.ndarray:\n            The transformed array which shape is specified by ``shape`` and\n            type will convert to complex if that of the input is another.\n\n    .. seealso:: :func:`scipy.fftpack.ifftn`\n\n    .. note::\n       The argument `plan` is currently experimental and the interface may be\n       changed in the future version.\n    '
    from cupy.cuda import cufft
    func = _default_fft_func(x, shape, axes, plan)
    return func(x, shape, axes, None, cufft.CUFFT_INVERSE, overwrite_x=overwrite_x, plan=plan)

def rfft(x, n=None, axis=-1, overwrite_x=False, plan=None):
    if False:
        return 10
    "Compute the one-dimensional FFT for real input.\n\n    The returned real array contains\n\n    .. code-block:: python\n\n        [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2))]  # if n is even\n        [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2)),Im(y(n/2))]  # if n is odd\n\n    Args:\n        x (cupy.ndarray): Array to be transformed.\n        n (None or int): Length of the transformed axis of the output. If ``n``\n            is not given, the length of the input along the axis specified by\n            ``axis`` is used.\n        axis (int): Axis over which to compute the FFT.\n        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.\n        plan (:class:`cupy.cuda.cufft.Plan1d` or ``None``): a cuFFT plan for\n            transforming ``x`` over ``axis``, which can be obtained using::\n\n                plan = cupyx.scipy.fftpack.get_fft_plan(\n                    x, axes, value_type='R2C')\n\n            Note that `plan` is defaulted to None, meaning CuPy will either\n            use an auto-generated plan behind the scene if cupy.fft.config.\n            enable_nd_planning = True, or use no cuFFT plan if it is set to\n            False.\n\n    Returns:\n        cupy.ndarray:\n            The transformed array.\n\n    .. seealso:: :func:`scipy.fftpack.rfft`\n\n    .. note::\n       The argument `plan` is currently experimental and the interface may be\n       changed in the future version.\n    "
    from cupy.cuda import cufft
    if n is None:
        n = x.shape[axis]
    shape = list(x.shape)
    shape[axis] = n
    f = _fft(x, (n,), (axis,), None, cufft.CUFFT_FORWARD, 'R2C', overwrite_x=overwrite_x, plan=plan)
    z = cupy.empty(shape, f.real.dtype)
    slice_z = [slice(None)] * x.ndim
    slice_f = [slice(None)] * x.ndim
    slice_z[axis] = slice(1)
    slice_f[axis] = slice(1)
    z[tuple(slice_z)] = f[tuple(slice_f)].real
    slice_z[axis] = slice(1, None, 2)
    slice_f[axis] = slice(1, None)
    z[tuple(slice_z)] = f[tuple(slice_f)].real
    slice_z[axis] = slice(2, None, 2)
    slice_f[axis] = slice(1, n - f.shape[axis] + 1)
    z[tuple(slice_z)] = f[tuple(slice_f)].imag
    return z

def irfft(x, n=None, axis=-1, overwrite_x=False):
    if False:
        print('Hello World!')
    'Compute the one-dimensional inverse FFT for real input.\n\n    Args:\n        x (cupy.ndarray): Array to be transformed.\n        n (None or int): Length of the transformed axis of the output. If ``n``\n            is not given, the length of the input along the axis specified by\n            ``axis`` is used.\n        axis (int): Axis over which to compute the FFT.\n        overwrite_x (bool): If True, the contents of ``x`` can be destroyed.\n\n    Returns:\n        cupy.ndarray:\n            The transformed array.\n\n    .. seealso:: :func:`scipy.fftpack.irfft`\n\n    .. note::\n       This function does not support a precomputed `plan`. If you need this\n       capability, please consider using :func:`cupy.fft.irfft` or :func:`\n       cupyx.scipy.fft.irfft`.\n    '
    from cupy.cuda import cufft
    if n is None:
        n = x.shape[axis]
    m = min(n, x.shape[axis])
    shape = list(x.shape)
    shape[axis] = n // 2 + 1
    if x.dtype in (cupy.float16, cupy.float32):
        z = cupy.zeros(shape, dtype=cupy.complex64)
    else:
        z = cupy.zeros(shape, dtype=cupy.complex128)
    slice_x = [slice(None)] * x.ndim
    slice_z = [slice(None)] * x.ndim
    slice_x[axis] = slice(1)
    slice_z[axis] = slice(1)
    z[tuple(slice_z)].real = x[tuple(slice_x)]
    slice_x[axis] = slice(1, m, 2)
    slice_z[axis] = slice(1, m // 2 + 1)
    z[tuple(slice_z)].real = x[tuple(slice_x)]
    slice_x[axis] = slice(2, m, 2)
    slice_z[axis] = slice(1, (m + 1) // 2)
    z[tuple(slice_z)].imag = x[tuple(slice_x)]
    return _fft(z, (n,), (axis,), None, cufft.CUFFT_INVERSE, 'C2R', overwrite_x=overwrite_x)