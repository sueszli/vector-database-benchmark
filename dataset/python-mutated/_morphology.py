import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters

@cupy.memoize(for_each_device=True)
def _get_binary_erosion_kernel(w_shape, int_type, offsets, center_is_true, border_value, invert, masked, all_weights_nonzero):
    if False:
        print('Hello World!')
    if invert:
        border_value = int(not border_value)
        true_val = 0
        false_val = 1
    else:
        true_val = 1
        false_val = 0
    if masked:
        pre = '\n            bool mv = (bool)mask[i];\n            bool _in = (bool)x[i];\n            if (!mv) {{\n                y = cast<Y>(_in);\n                return;\n            }} else if ({center_is_true} && _in == {false_val}) {{\n                y = cast<Y>(_in);\n                return;\n            }}'.format(center_is_true=int(center_is_true), false_val=false_val)
    else:
        pre = '\n            bool _in = (bool)x[i];\n            if ({center_is_true} && _in == {false_val}) {{\n                y = cast<Y>(_in);\n                return;\n            }}'.format(center_is_true=int(center_is_true), false_val=false_val)
    pre = pre + '\n            y = cast<Y>({true_val});'.format(true_val=true_val)
    found = '\n        if ({{cond}}) {{{{\n            if (!{border_value}) {{{{\n                y = cast<Y>({false_val});\n                return;\n            }}}}\n        }}}} else {{{{\n            bool nn = {{value}} ? {true_val} : {false_val};\n            if (!nn) {{{{\n                y = cast<Y>({false_val});\n                return;\n            }}}}\n        }}}}'.format(true_val=int(true_val), false_val=int(false_val), border_value=int(border_value))
    name = 'binary_erosion'
    if false_val:
        name += '_invert'
    return _filters_core._generate_nd_kernel(name, pre, found, '', 'constant', w_shape, int_type, offsets, 0, ctype='Y', has_weights=True, has_structure=False, has_mask=masked, binary_morphology=True, all_weights_nonzero=all_weights_nonzero)

def _center_is_true(structure, origin):
    if False:
        while True:
            i = 10
    coor = tuple([oo + ss // 2 for (ss, oo) in zip(structure.shape, origin)])
    return bool(structure[coor])

def iterate_structure(structure, iterations, origin=None):
    if False:
        print('Hello World!')
    'Iterate a structure by dilating it with itself.\n\n    Args:\n        structure(array_like): Structuring element (an array of bools,\n            for example), to be dilated with itself.\n        iterations(int): The number of dilations performed on the structure\n            with itself.\n        origin(int or tuple of int, optional): If origin is None, only the\n            iterated structure is returned. If not, a tuple of the iterated\n            structure and the modified origin is returned.\n\n    Returns:\n        cupy.ndarray: A new structuring element obtained by dilating\n        ``structure`` (``iterations`` - 1) times with itself.\n\n    .. seealso:: :func:`scipy.ndimage.iterate_structure`\n    '
    if iterations < 2:
        return structure.copy()
    ni = iterations - 1
    shape = [ii + ni * (ii - 1) for ii in structure.shape]
    pos = [ni * (structure.shape[ii] // 2) for ii in range(len(shape))]
    slc = tuple((slice(pos[ii], pos[ii] + structure.shape[ii], None) for ii in range(len(shape))))
    out = cupy.zeros(shape, bool)
    out[slc] = structure != 0
    out = binary_dilation(out, structure, iterations=ni)
    if origin is None:
        return out
    else:
        origin = _util._fix_sequence_arg(origin, structure.ndim, 'origin', int)
        origin = [iterations * o for o in origin]
        return (out, origin)

def generate_binary_structure(rank, connectivity):
    if False:
        for i in range(10):
            print('nop')
    'Generate a binary structure for binary morphological operations.\n\n    Args:\n        rank(int): Number of dimensions of the array to which the structuring\n            element will be applied, as returned by ``np.ndim``.\n        connectivity(int): ``connectivity`` determines which elements of the\n            output array belong to the structure, i.e., are considered as\n            neighbors of the central element. Elements up to a squared distance\n            of ``connectivity`` from the center are considered neighbors.\n            ``connectivity`` may range from 1 (no diagonal elements are\n            neighbors) to ``rank`` (all elements are neighbors).\n\n    Returns:\n        cupy.ndarray: Structuring element which may be used for binary\n        morphological operations, with ``rank`` dimensions and all\n        dimensions equal to 3.\n\n    .. seealso:: :func:`scipy.ndimage.generate_binary_structure`\n    '
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return cupy.asarray(True, dtype=bool)
    output = numpy.fabs(numpy.indices([3] * rank) - 1)
    output = numpy.add.reduce(output, 0)
    output = output <= connectivity
    return cupy.asarray(output)

def _binary_erosion(input, structure, iterations, mask, output, border_value, origin, invert, brute_force=True):
    if False:
        print('Hello World!')
    try:
        iterations = operator.index(iterations)
    except TypeError:
        raise TypeError('iterations parameter should be an integer')
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
        all_weights_nonzero = input.ndim == 1
        center_is_true = True
        default_structure = True
    else:
        structure = structure.astype(dtype=bool, copy=False)
        default_structure = False
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have same dimensionality')
    if not structure.flags.c_contiguous:
        structure = cupy.ascontiguousarray(structure)
    if structure.size < 1:
        raise RuntimeError('structure must not be empty')
    if mask is not None:
        if mask.shape != input.shape:
            raise RuntimeError('mask and input must have equal sizes')
        if not mask.flags.c_contiguous:
            mask = cupy.ascontiguousarray(mask)
        masked = True
    else:
        masked = False
    origin = _util._fix_sequence_arg(origin, input.ndim, 'origin', int)
    if isinstance(output, cupy.ndarray):
        if output.dtype.kind == 'c':
            raise TypeError('Complex output type not supported')
    else:
        output = bool
    output = _util._get_output(output, input)
    temp_needed = cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS')
    if temp_needed:
        temp = output
        output = _util._get_output(output.dtype, input)
    if structure.ndim == 0:
        if float(structure):
            output[...] = cupy.asarray(input, dtype=bool)
        else:
            output[...] = ~cupy.asarray(input, dtype=bool)
        return output
    origin = tuple(origin)
    int_type = _util._get_inttype(input)
    offsets = _filters_core._origins_to_offsets(origin, structure.shape)
    if not default_structure:
        nnz = int(cupy.count_nonzero(structure))
        all_weights_nonzero = nnz == structure.size
        if all_weights_nonzero:
            center_is_true = True
        else:
            center_is_true = _center_is_true(structure, origin)
    erode_kernel = _get_binary_erosion_kernel(structure.shape, int_type, offsets, center_is_true, border_value, invert, masked, all_weights_nonzero)
    if iterations == 1:
        if masked:
            output = erode_kernel(input, structure, mask, output)
        else:
            output = erode_kernel(input, structure, output)
    elif center_is_true and (not brute_force):
        raise NotImplementedError('only brute_force iteration has been implemented')
    else:
        if cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS'):
            raise ValueError('output and input may not overlap in memory')
        tmp_in = cupy.empty_like(input, dtype=output.dtype)
        tmp_out = output
        if iterations >= 1 and (not iterations & 1):
            (tmp_in, tmp_out) = (tmp_out, tmp_in)
        if masked:
            tmp_out = erode_kernel(input, structure, mask, tmp_out)
        else:
            tmp_out = erode_kernel(input, structure, tmp_out)
        changed = not (input == tmp_out).all()
        ii = 1
        while ii < iterations or (iterations < 1 and changed):
            (tmp_in, tmp_out) = (tmp_out, tmp_in)
            if masked:
                tmp_out = erode_kernel(tmp_in, structure, mask, tmp_out)
            else:
                tmp_out = erode_kernel(tmp_in, structure, tmp_out)
            changed = not (tmp_in == tmp_out).all()
            ii += 1
            if not changed and (not ii & 1):
                break
        output = tmp_out
    if temp_needed:
        _core.elementwise_copy(output, temp)
        output = temp
    return output

def binary_erosion(input, structure=None, iterations=1, mask=None, output=None, border_value=0, origin=0, brute_force=False):
    if False:
        while True:
            i = 10
    'Multidimensional binary erosion with a given structuring element.\n\n    Binary erosion is a mathematical morphology operation used for image\n    processing.\n\n    Args:\n        input(cupy.ndarray): The input binary array_like to be eroded.\n            Non-zero (True) elements form the subset to be eroded.\n        structure(cupy.ndarray, optional): The structuring element used for the\n            erosion. Non-zero elements are considered True. If no structuring\n            element is provided an element is generated with a square\n            connectivity equal to one. (Default value = None).\n        iterations(int, optional): The erosion is repeated ``iterations`` times\n            (one, by default). If iterations is less than 1, the erosion is\n            repeated until the result does not change anymore. Only an integer\n            of iterations is accepted.\n        mask(cupy.ndarray or None, optional): If a mask is given, only those\n            elements with a True value at the corresponding mask element are\n            modified at each iteration. (Default value = None)\n        output(cupy.ndarray, optional): Array of the same shape as input, into\n            which the output is placed. By default, a new array is created.\n        border_value(int (cast to 0 or 1), optional): Value at the\n            border in the output array. (Default value = 0)\n        origin(int or tuple of ints, optional): Placement of the filter, by\n            default 0.\n        brute_force(boolean, optional): Memory condition: if False, only the\n            pixels whose value was changed in the last iteration are tracked as\n            candidates to be updated (eroded) in the current iteration; if\n            True all pixels are considered as candidates for erosion,\n            regardless of what happened in the previous iteration.\n\n    Returns:\n        cupy.ndarray: The result of binary erosion.\n\n    .. warning::\n\n        This function may synchronize the device.\n\n    .. seealso:: :func:`scipy.ndimage.binary_erosion`\n    '
    return _binary_erosion(input, structure, iterations, mask, output, border_value, origin, 0, brute_force)

def binary_dilation(input, structure=None, iterations=1, mask=None, output=None, border_value=0, origin=0, brute_force=False):
    if False:
        return 10
    'Multidimensional binary dilation with the given structuring element.\n\n    Args:\n        input(cupy.ndarray): The input binary array_like to be dilated.\n            Non-zero (True) elements form the subset to be dilated.\n        structure(cupy.ndarray, optional): The structuring element used for the\n            dilation. Non-zero elements are considered True. If no structuring\n            element is provided an element is generated with a square\n            connectivity equal to one. (Default value = None).\n        iterations(int, optional): The dilation is repeated ``iterations``\n            times (one, by default). If iterations is less than 1, the dilation\n            is repeated until the result does not change anymore. Only an\n            integer of iterations is accepted.\n        mask(cupy.ndarray or None, optional): If a mask is given, only those\n            elements with a True value at the corresponding mask element are\n            modified at each iteration. (Default value = None)\n        output(cupy.ndarray, optional): Array of the same shape as input, into\n            which the output is placed. By default, a new array is created.\n        border_value(int (cast to 0 or 1), optional): Value at the\n            border in the output array. (Default value = 0)\n        origin(int or tuple of ints, optional): Placement of the filter, by\n            default 0.\n        brute_force(boolean, optional): Memory condition: if False, only the\n            pixels whose value was changed in the last iteration are tracked as\n            candidates to be updated (dilated) in the current iteration; if\n            True all pixels are considered as candidates for dilation,\n            regardless of what happened in the previous iteration.\n\n    Returns:\n        cupy.ndarray: The result of binary dilation.\n\n    .. warning::\n\n        This function may synchronize the device.\n\n    .. seealso:: :func:`scipy.ndimage.binary_dilation`\n    '
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    origin = _util._fix_sequence_arg(origin, input.ndim, 'origin', int)
    structure = structure[tuple([slice(None, None, -1)] * structure.ndim)]
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure.shape[ii] & 1:
            origin[ii] -= 1
    return _binary_erosion(input, structure, iterations, mask, output, border_value, origin, 1, brute_force)

def binary_opening(input, structure=None, iterations=1, output=None, origin=0, mask=None, border_value=0, brute_force=False):
    if False:
        return 10
    '\n    Multidimensional binary opening with the given structuring element.\n\n    The *opening* of an input image by a structuring element is the\n    *dilation* of the *erosion* of the image by the structuring element.\n\n    Args:\n        input(cupy.ndarray): The input binary array to be opened.\n            Non-zero (True) elements form the subset to be opened.\n        structure(cupy.ndarray, optional): The structuring element used for the\n            opening. Non-zero elements are considered True. If no structuring\n            element is provided an element is generated with a square\n            connectivity equal to one. (Default value = None).\n        iterations(int, optional): The opening is repeated ``iterations`` times\n            (one, by default). If iterations is less than 1, the opening is\n            repeated until the result does not change anymore. Only an integer\n            of iterations is accepted.\n        output(cupy.ndarray, optional): Array of the same shape as input, into\n            which the output is placed. By default, a new array is created.\n        origin(int or tuple of ints, optional): Placement of the filter, by\n            default 0.\n        mask(cupy.ndarray or None, optional): If a mask is given, only those\n            elements with a True value at the corresponding mask element are\n            modified at each iteration. (Default value = None)\n        border_value(int (cast to 0 or 1), optional): Value at the\n            border in the output array. (Default value = 0)\n        brute_force(boolean, optional): Memory condition: if False, only the\n            pixels whose value was changed in the last iteration are tracked as\n            candidates to be updated (dilated) in the current iteration; if\n            True all pixels are considered as candidates for opening,\n            regardless of what happened in the previous iteration.\n\n    Returns:\n        cupy.ndarray: The result of binary opening.\n\n    .. warning::\n\n        This function may synchronize the device.\n\n    .. seealso:: :func:`scipy.ndimage.binary_opening`\n    '
    if structure is None:
        rank = input.ndim
        structure = generate_binary_structure(rank, 1)
    tmp = binary_erosion(input, structure, iterations, mask, None, border_value, origin, brute_force)
    return binary_dilation(tmp, structure, iterations, mask, output, border_value, origin, brute_force)

def binary_closing(input, structure=None, iterations=1, output=None, origin=0, mask=None, border_value=0, brute_force=False):
    if False:
        while True:
            i = 10
    '\n    Multidimensional binary closing with the given structuring element.\n\n    The *closing* of an input image by a structuring element is the\n    *erosion* of the *dilation* of the image by the structuring element.\n\n    Args:\n        input(cupy.ndarray): The input binary array to be closed.\n            Non-zero (True) elements form the subset to be closed.\n        structure(cupy.ndarray, optional): The structuring element used for the\n            closing. Non-zero elements are considered True. If no structuring\n            element is provided an element is generated with a square\n            connectivity equal to one. (Default value = None).\n        iterations(int, optional): The closing is repeated ``iterations`` times\n            (one, by default). If iterations is less than 1, the closing is\n            repeated until the result does not change anymore. Only an integer\n            of iterations is accepted.\n        output(cupy.ndarray, optional): Array of the same shape as input, into\n            which the output is placed. By default, a new array is created.\n        origin(int or tuple of ints, optional): Placement of the filter, by\n            default 0.\n        mask(cupy.ndarray or None, optional): If a mask is given, only those\n            elements with a True value at the corresponding mask element are\n            modified at each iteration. (Default value = None)\n        border_value(int (cast to 0 or 1), optional): Value at the\n            border in the output array. (Default value = 0)\n        brute_force(boolean, optional): Memory condition: if False, only the\n            pixels whose value was changed in the last iteration are tracked as\n            candidates to be updated (dilated) in the current iteration; if\n            True all pixels are considered as candidates for closing,\n            regardless of what happened in the previous iteration.\n\n    Returns:\n        cupy.ndarray: The result of binary closing.\n\n    .. warning::\n\n        This function may synchronize the device.\n\n    .. seealso:: :func:`scipy.ndimage.binary_closing`\n    '
    if structure is None:
        rank = input.ndim
        structure = generate_binary_structure(rank, 1)
    tmp = binary_dilation(input, structure, iterations, mask, None, border_value, origin, brute_force)
    return binary_erosion(tmp, structure, iterations, mask, output, border_value, origin, brute_force)

def binary_hit_or_miss(input, structure1=None, structure2=None, output=None, origin1=0, origin2=None):
    if False:
        print('Hello World!')
    '\n    Multidimensional binary hit-or-miss transform.\n\n    The hit-or-miss transform finds the locations of a given pattern\n    inside the input image.\n\n    Args:\n        input (cupy.ndarray): Binary image where a pattern is to be detected.\n        structure1 (cupy.ndarray, optional): Part of the structuring element to\n            be fitted to the foreground (non-zero elements) of ``input``. If no\n            value is provided, a structure of square connectivity 1 is chosen.\n        structure2 (cupy.ndarray, optional): Second part of the structuring\n            element that has to miss completely the foreground. If no value is\n            provided, the complementary of ``structure1`` is taken.\n        output (cupy.ndarray, dtype or None, optional): Array of the same shape\n            as input, into which the output is placed. By default, a new array\n            is created.\n        origin1 (int or tuple of ints, optional): Placement of the first part\n            of the structuring element ``structure1``, by default 0 for a\n            centered structure.\n        origin2 (int or tuple of ints or None, optional): Placement of the\n            second part of the structuring element ``structure2``, by default 0\n            for a centered structure. If a value is provided for ``origin1``\n            and not for ``origin2``, then ``origin2`` is set to ``origin1``.\n\n    Returns:\n        cupy.ndarray: Hit-or-miss transform of ``input`` with the given\n        structuring element (``structure1``, ``structure2``).\n\n    .. warning::\n\n        This function may synchronize the device.\n\n    .. seealso:: :func:`scipy.ndimage.binary_hit_or_miss`\n    '
    if structure1 is None:
        structure1 = generate_binary_structure(input.ndim, 1)
    if structure2 is None:
        structure2 = cupy.logical_not(structure1)
    origin1 = _util._fix_sequence_arg(origin1, input.ndim, 'origin1', int)
    if origin2 is None:
        origin2 = origin1
    else:
        origin2 = _util._fix_sequence_arg(origin2, input.ndim, 'origin2', int)
    tmp1 = _binary_erosion(input, structure1, 1, None, None, 0, origin1, 0, False)
    inplace = isinstance(output, cupy.ndarray)
    result = _binary_erosion(input, structure2, 1, None, output, 0, origin2, 1, False)
    if inplace:
        cupy.logical_not(output, output)
        cupy.logical_and(tmp1, output, output)
    else:
        cupy.logical_not(result, result)
        return cupy.logical_and(tmp1, result)

def binary_propagation(input, structure=None, mask=None, output=None, border_value=0, origin=0):
    if False:
        i = 10
        return i + 15
    '\n    Multidimensional binary propagation with the given structuring element.\n\n    Args:\n        input (cupy.ndarray): Binary image to be propagated inside ``mask``.\n        structure (cupy.ndarray, optional): Structuring element used in the\n            successive dilations. The output may depend on the structuring\n            element, especially if ``mask`` has several connex components. If\n            no structuring element is provided, an element is generated with a\n            squared connectivity equal to one.\n        mask (cupy.ndarray, optional): Binary mask defining the region into\n            which ``input`` is allowed to propagate.\n        output (cupy.ndarray, optional): Array of the same shape as input, into\n            which the output is placed. By default, a new array is created.\n        border_value (int, optional): Value at the border in the output array.\n            The value is cast to 0 or 1.\n        origin (int or tuple of ints, optional): Placement of the filter.\n\n    Returns:\n        cupy.ndarray : Binary propagation of ``input`` inside ``mask``.\n\n    .. warning::\n\n        This function may synchronize the device.\n\n    .. seealso:: :func:`scipy.ndimage.binary_propagation`\n    '
    return binary_dilation(input, structure, -1, mask, output, border_value, origin, brute_force=True)

def binary_fill_holes(input, structure=None, output=None, origin=0):
    if False:
        while True:
            i = 10
    'Fill the holes in binary objects.\n\n    Args:\n        input (cupy.ndarray): N-D binary array with holes to be filled.\n        structure (cupy.ndarray, optional):  Structuring element used in the\n            computation; large-size elements make computations faster but may\n            miss holes separated from the background by thin regions. The\n            default element (with a square connectivity equal to one) yields\n            the intuitive result where all holes in the input have been filled.\n        output (cupy.ndarray, dtype or None, optional): Array of the same shape\n            as input, into which the output is placed. By default, a new array\n            is created.\n        origin (int, tuple of ints, optional): Position of the structuring\n            element.\n\n    Returns:\n        cupy.ndarray: Transformation of the initial image ``input`` where holes\n        have been filled.\n\n    .. warning::\n\n        This function may synchronize the device.\n\n    .. seealso:: :func:`scipy.ndimage.binary_fill_holes`\n    '
    mask = cupy.logical_not(input)
    tmp = cupy.zeros(mask.shape, bool)
    inplace = isinstance(output, cupy.ndarray)
    if inplace:
        binary_dilation(tmp, structure, -1, mask, output, 1, origin, brute_force=True)
        cupy.logical_not(output, output)
    else:
        output = binary_dilation(tmp, structure, -1, mask, None, 1, origin, brute_force=True)
        cupy.logical_not(output, output)
        return output

def grey_erosion(input, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        return 10
    "Calculates a greyscale erosion.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (tuple of ints): Shape of a flat and full structuring element used\n            for the greyscale erosion. Optional if ``footprint`` or\n            ``structure`` is provided.\n        footprint (array of ints): Positions of non-infinite elements of a flat\n            structuring element used for greyscale erosion. Non-zero values\n            give the set of neighbors of the center over which minimum is\n            chosen.\n        structure (array of ints): Structuring element used for the greyscale\n            erosion. ``structure`` may be a non-flat structuring element.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``constant``. Default is ``0.0``.\n        origin (scalar or tuple of scalar): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of greyscale erosion.\n\n    .. seealso:: :func:`scipy.ndimage.grey_erosion`\n    "
    if size is None and footprint is None and (structure is None):
        raise ValueError('size, footprint or structure must be specified')
    return _filters._min_or_max_filter(input, size, footprint, structure, output, mode, cval, origin, 'min')

def grey_dilation(input, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        print('Hello World!')
    "Calculates a greyscale dilation.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (tuple of ints): Shape of a flat and full structuring element used\n            for the greyscale dilation. Optional if ``footprint`` or\n            ``structure`` is provided.\n        footprint (array of ints): Positions of non-infinite elements of a flat\n            structuring element used for greyscale dilation. Non-zero values\n            give the set of neighbors of the center over which maximum is\n            chosen.\n        structure (array of ints): Structuring element used for the greyscale\n            dilation. ``structure`` may be a non-flat structuring element.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``constant``. Default is ``0.0``.\n        origin (scalar or tuple of scalar): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of greyscale dilation.\n\n    .. seealso:: :func:`scipy.ndimage.grey_dilation`\n    "
    if size is None and footprint is None and (structure is None):
        raise ValueError('size, footprint or structure must be specified')
    if structure is not None:
        structure = cupy.array(structure)
        structure = structure[tuple([slice(None, None, -1)] * structure.ndim)]
    if footprint is not None:
        footprint = cupy.array(footprint)
        footprint = footprint[tuple([slice(None, None, -1)] * footprint.ndim)]
    origin = _util._fix_sequence_arg(origin, input.ndim, 'origin', int)
    for i in range(len(origin)):
        origin[i] = -origin[i]
        if footprint is not None:
            sz = footprint.shape[i]
        elif structure is not None:
            sz = structure.shape[i]
        elif numpy.isscalar(size):
            sz = size
        else:
            sz = size[i]
        if sz % 2 == 0:
            origin[i] -= 1
    return _filters._min_or_max_filter(input, size, footprint, structure, output, mode, cval, origin, 'max')

def grey_closing(input, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        return 10
    "Calculates a multi-dimensional greyscale closing.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (tuple of ints): Shape of a flat and full structuring element used\n            for the greyscale closing. Optional if ``footprint`` or\n            ``structure`` is provided.\n        footprint (array of ints): Positions of non-infinite elements of a flat\n            structuring element used for greyscale closing. Non-zero values\n            give the set of neighbors of the center over which closing is\n            chosen.\n        structure (array of ints): Structuring element used for the greyscale\n            closing. ``structure`` may be a non-flat structuring element.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``constant``. Default is ``0.0``.\n        origin (scalar or tuple of scalar): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of greyscale closing.\n\n    .. seealso:: :func:`scipy.ndimage.grey_closing`\n    "
    if size is not None and footprint is not None:
        warnings.warn('ignoring size because footprint is set', UserWarning, stacklevel=2)
    tmp = grey_dilation(input, size, footprint, structure, None, mode, cval, origin)
    return grey_erosion(tmp, size, footprint, structure, output, mode, cval, origin)

def grey_opening(input, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        print('Hello World!')
    "Calculates a multi-dimensional greyscale opening.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (tuple of ints): Shape of a flat and full structuring element used\n            for the greyscale opening. Optional if ``footprint`` or\n            ``structure`` is provided.\n        footprint (array of ints): Positions of non-infinite elements of a flat\n            structuring element used for greyscale opening. Non-zero values\n            give the set of neighbors of the center over which opening is\n            chosen.\n        structure (array of ints): Structuring element used for the greyscale\n            opening. ``structure`` may be a non-flat structuring element.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``constant``. Default is ``0.0``.\n        origin (scalar or tuple of scalar): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The result of greyscale opening.\n\n    .. seealso:: :func:`scipy.ndimage.grey_opening`\n    "
    if size is not None and footprint is not None:
        warnings.warn('ignoring size because footprint is set', UserWarning, stacklevel=2)
    tmp = grey_erosion(input, size, footprint, structure, None, mode, cval, origin)
    return grey_dilation(tmp, size, footprint, structure, output, mode, cval, origin)

def morphological_gradient(input, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        print('Hello World!')
    "\n    Multidimensional morphological gradient.\n\n    The morphological gradient is calculated as the difference between a\n    dilation and an erosion of the input with a given structuring element.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (tuple of ints): Shape of a flat and full structuring element used\n            for the morphological gradient. Optional if ``footprint`` or\n            ``structure`` is provided.\n        footprint (array of ints): Positions of non-infinite elements of a flat\n            structuring element used for morphological gradient. Non-zero\n            values give the set of neighbors of the center over which opening\n            is chosen.\n        structure (array of ints): Structuring element used for the\n            morphological gradient. ``structure`` may be a non-flat\n            structuring element.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``constant``. Default is ``0.0``.\n        origin (scalar or tuple of scalar): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The morphological gradient of the input.\n\n    .. seealso:: :func:`scipy.ndimage.morphological_gradient`\n    "
    tmp = grey_dilation(input, size, footprint, structure, None, mode, cval, origin)
    if isinstance(output, cupy.ndarray):
        grey_erosion(input, size, footprint, structure, output, mode, cval, origin)
        return cupy.subtract(tmp, output, output)
    else:
        return tmp - grey_erosion(input, size, footprint, structure, None, mode, cval, origin)

def morphological_laplace(input, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        return 10
    "\n    Multidimensional morphological laplace.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (tuple of ints): Shape of a flat and full structuring element used\n            for the morphological laplace. Optional if ``footprint`` or\n            ``structure`` is provided.\n        footprint (array of ints): Positions of non-infinite elements of a flat\n            structuring element used for morphological laplace. Non-zero\n            values give the set of neighbors of the center over which opening\n            is chosen.\n        structure (array of ints): Structuring element used for the\n            morphological laplace. ``structure`` may be a non-flat\n            structuring element.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``constant``. Default is ``0.0``.\n        origin (scalar or tuple of scalar): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: The morphological laplace of the input.\n\n    .. seealso:: :func:`scipy.ndimage.morphological_laplace`\n    "
    tmp1 = grey_dilation(input, size, footprint, structure, None, mode, cval, origin)
    if isinstance(output, cupy.ndarray):
        grey_erosion(input, size, footprint, structure, output, mode, cval, origin)
        cupy.add(tmp1, output, output)
        cupy.subtract(output, input, output)
        return cupy.subtract(output, input, output)
    else:
        tmp2 = grey_erosion(input, size, footprint, structure, None, mode, cval, origin)
        cupy.add(tmp1, tmp2, tmp2)
        cupy.subtract(tmp2, input, tmp2)
        cupy.subtract(tmp2, input, tmp2)
        return tmp2

def white_tophat(input, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        return 10
    "\n    Multidimensional white tophat filter.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (tuple of ints): Shape of a flat and full structuring element used\n            for the white tophat. Optional if ``footprint`` or ``structure`` is\n            provided.\n        footprint (array of ints): Positions of non-infinite elements of a flat\n            structuring element used for the white tophat. Non-zero values\n            give the set of neighbors of the center over which opening is\n            chosen.\n        structure (array of ints): Structuring element used for the white\n            tophat. ``structure`` may be a non-flat structuring element.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``constant``. Default is ``0.0``.\n        origin (scalar or tuple of scalar): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarray: Result of the filter of ``input`` with ``structure``.\n\n    .. seealso:: :func:`scipy.ndimage.white_tophat`\n    "
    if size is not None and footprint is not None:
        warnings.warn('ignoring size because footprint is set', UserWarning, stacklevel=2)
    tmp = grey_erosion(input, size, footprint, structure, None, mode, cval, origin)
    tmp = grey_dilation(tmp, size, footprint, structure, output, mode, cval, origin)
    if input.dtype == numpy.bool_ and tmp.dtype == numpy.bool_:
        cupy.bitwise_xor(input, tmp, out=tmp)
    else:
        cupy.subtract(input, tmp, out=tmp)
    return tmp

def black_tophat(input, size=None, footprint=None, structure=None, output=None, mode='reflect', cval=0.0, origin=0):
    if False:
        print('Hello World!')
    "\n    Multidimensional black tophat filter.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        size (tuple of ints): Shape of a flat and full structuring element used\n            for the black tophat. Optional if ``footprint`` or ``structure`` is\n            provided.\n        footprint (array of ints): Positions of non-infinite elements of a flat\n            structuring element used for the black tophat. Non-zero values\n            give the set of neighbors of the center over which opening is\n            chosen.\n        structure (array of ints): Structuring element used for the black\n            tophat. ``structure`` may be a non-flat structuring element.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output.\n        mode (str): The array borders are handled according to the given mode\n            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,\n            ``'wrap'``). Default is ``'reflect'``.\n        cval (scalar): Value to fill past edges of input if mode is\n            ``constant``. Default is ``0.0``.\n        origin (scalar or tuple of scalar): The origin parameter controls the\n            placement of the filter, relative to the center of the current\n            element of the input. Default of 0 is equivalent to\n            ``(0,)*input.ndim``.\n\n    Returns:\n        cupy.ndarry : Result of the filter of ``input`` with ``structure``.\n\n    .. seealso:: :func:`scipy.ndimage.black_tophat`\n    "
    if size is not None and footprint is not None:
        warnings.warn('ignoring size because footprint is set', UserWarning, stacklevel=2)
    tmp = grey_dilation(input, size, footprint, structure, None, mode, cval, origin)
    tmp = grey_erosion(tmp, size, footprint, structure, output, mode, cval, origin)
    if input.dtype == numpy.bool_ and tmp.dtype == numpy.bool_:
        cupy.bitwise_xor(tmp, input, out=tmp)
    else:
        cupy.subtract(tmp, input, out=tmp)
    return tmp