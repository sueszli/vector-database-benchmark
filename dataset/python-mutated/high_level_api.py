import abc
from collections import OrderedDict, defaultdict
import numpy as np
from .utils import deserialize_class
__all__ = ['BaseHighLevelWCS', 'HighLevelWCSMixin']

def rec_getattr(obj, att):
    if False:
        print('Hello World!')
    for a in att.split('.'):
        obj = getattr(obj, a)
    return obj

def default_order(components):
    if False:
        while True:
            i = 10
    order = []
    for (key, _, _) in components:
        if key not in order:
            order.append(key)
    return order

def _toindex(value):
    if False:
        for i in range(10):
            print('nop')
    'Convert value to an int or an int array.\n\n    Input coordinates converted to integers\n    corresponding to the center of the pixel.\n    The convention is that the center of the pixel is\n    (0, 0), while the lower left corner is (-0.5, -0.5).\n    The outputs are used to index the mask.\n\n    Examples\n    --------\n    >>> _toindex(np.array([-0.5, 0.49999]))\n    array([0, 0])\n    >>> _toindex(np.array([0.5, 1.49999]))\n    array([1, 1])\n    >>> _toindex(np.array([1.5, 2.49999]))\n    array([2, 2])\n    '
    indx = np.asarray(np.floor(np.asarray(value) + 0.5), dtype=int)
    return indx

class BaseHighLevelWCS(metaclass=abc.ABCMeta):
    """
    Abstract base class for the high-level WCS interface.

    This is described in `APE 14: A shared Python interface for World Coordinate
    Systems <https://doi.org/10.5281/zenodo.1188875>`_.
    """

    @property
    @abc.abstractmethod
    def low_level_wcs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a reference to the underlying low-level WCS object.\n        '

    @abc.abstractmethod
    def pixel_to_world(self, *pixel_arrays):
        if False:
            i = 10
            return i + 15
        '\n        Convert pixel coordinates to world coordinates (represented by\n        high-level objects).\n\n        If a single high-level object is used to represent the world coordinates\n        (i.e., if ``len(wcs.world_axis_object_classes) == 1``), it is returned\n        as-is (not in a tuple/list), otherwise a tuple of high-level objects is\n        returned. See\n        `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_to_world_values` for pixel\n        indexing and ordering conventions.\n        '

    def array_index_to_world(self, *index_arrays):
        if False:
            return 10
        '\n        Convert array indices to world coordinates (represented by Astropy\n        objects).\n\n        If a single high-level object is used to represent the world coordinates\n        (i.e., if ``len(wcs.world_axis_object_classes) == 1``), it is returned\n        as-is (not in a tuple/list), otherwise a tuple of high-level objects is\n        returned. See\n        `~astropy.wcs.wcsapi.BaseLowLevelWCS.array_index_to_world_values` for\n        pixel indexing and ordering conventions.\n        '
        return self.pixel_to_world(*index_arrays[::-1])

    @abc.abstractmethod
    def world_to_pixel(self, *world_objects):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert world coordinates (represented by Astropy objects) to pixel\n        coordinates.\n\n        If `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim` is ``1``, this\n        method returns a single scalar or array, otherwise a tuple of scalars or\n        arrays is returned. See\n        `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_to_pixel_values` for pixel\n        indexing and ordering conventions.\n        '

    def world_to_array_index(self, *world_objects):
        if False:
            while True:
                i = 10
        '\n        Convert world coordinates (represented by Astropy objects) to array\n        indices.\n\n        If `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim` is ``1``, this\n        method returns a single scalar or array, otherwise a tuple of scalars or\n        arrays is returned. See\n        `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_to_array_index_values` for\n        pixel indexing and ordering conventions. The indices should be returned\n        as rounded integers.\n        '
        if self.low_level_wcs.pixel_n_dim == 1:
            return _toindex(self.world_to_pixel(*world_objects))
        else:
            return tuple(_toindex(self.world_to_pixel(*world_objects)[::-1]).tolist())

def high_level_objects_to_values(*world_objects, low_level_wcs):
    if False:
        while True:
            i = 10
    '\n    Convert the input high level object to low level values.\n\n    This function uses the information in ``wcs.world_axis_object_classes`` and\n    ``wcs.world_axis_object_components`` to convert the high level objects\n    (such as `~.SkyCoord`) to low level "values" `~.Quantity` objects.\n\n    This is used in `.HighLevelWCSMixin.world_to_pixel`, but provided as a\n    separate function for use in other places where needed.\n\n    Parameters\n    ----------\n    *world_objects: object\n        High level coordinate objects.\n\n    low_level_wcs: `.BaseLowLevelWCS`\n        The WCS object to use to interpret the coordinates.\n    '
    serialized_classes = low_level_wcs.world_axis_object_classes
    components = low_level_wcs.world_axis_object_components
    classes = OrderedDict()
    for key in default_order(components):
        if low_level_wcs.serialized_classes:
            classes[key] = deserialize_class(serialized_classes[key], construct=False)
        else:
            classes[key] = serialized_classes[key]
    if len(world_objects) != len(classes):
        raise ValueError(f'Number of world inputs ({len(world_objects)}) does not match expected ({len(classes)})')
    world_by_key = {}
    unique_match = True
    for w in world_objects:
        matches = []
        for (key, (klass, *_)) in classes.items():
            if isinstance(w, klass):
                matches.append(key)
        if len(matches) == 1:
            world_by_key[matches[0]] = w
        else:
            unique_match = False
            break
    objects = {}
    if unique_match:
        for (key, (klass, args, kwargs, *rest)) in classes.items():
            if len(rest) == 0:
                klass_gen = klass
            elif len(rest) == 1:
                klass_gen = rest[0]
            else:
                raise ValueError('Tuples in world_axis_object_classes should have length 3 or 4')
            from astropy.coordinates import SkyCoord
            if isinstance(world_by_key[key], SkyCoord):
                if 'frame' in kwargs:
                    objects[key] = world_by_key[key].transform_to(kwargs['frame'])
                else:
                    objects[key] = world_by_key[key]
            else:
                objects[key] = klass_gen(world_by_key[key], *args, **kwargs)
    else:
        for (ikey, key) in enumerate(classes):
            (klass, args, kwargs, *rest) = classes[key]
            if len(rest) == 0:
                klass_gen = klass
            elif len(rest) == 1:
                klass_gen = rest[0]
            else:
                raise ValueError('Tuples in world_axis_object_classes should have length 3 or 4')
            w = world_objects[ikey]
            if not isinstance(w, klass):
                raise ValueError(f"Expected the following order of world arguments: {', '.join([k.__name__ for (k, *_) in classes.values()])}")
            from astropy.coordinates import SkyCoord
            if isinstance(w, SkyCoord):
                if 'frame' in kwargs:
                    objects[key] = w.transform_to(kwargs['frame'])
                else:
                    objects[key] = w
            else:
                objects[key] = klass_gen(w, *args, **kwargs)
    world = []
    for (key, _, attr) in components:
        if callable(attr):
            world.append(attr(objects[key]))
        else:
            world.append(rec_getattr(objects[key], attr))
    return world

def values_to_high_level_objects(*world_values, low_level_wcs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert low level values into high level objects.\n\n    This function uses the information in ``wcs.world_axis_object_classes`` and\n    ``wcs.world_axis_object_components`` to convert low level "values"\n    `~.Quantity` objects, to high level objects (such as `~.SkyCoord).\n\n    This is used in `.HighLevelWCSMixin.pixel_to_world`, but provided as a\n    separate function for use in other places where needed.\n\n    Parameters\n    ----------\n    *world_values: object\n        Low level, "values" representations of the world coordinates.\n\n    low_level_wcs: `.BaseLowLevelWCS`\n        The WCS object to use to interpret the coordinates.\n    '
    components = low_level_wcs.world_axis_object_components
    classes = low_level_wcs.world_axis_object_classes
    if low_level_wcs.serialized_classes:
        classes_new = {}
        for (key, value) in classes.items():
            classes_new[key] = deserialize_class(value, construct=False)
        classes = classes_new
    args = defaultdict(list)
    kwargs = defaultdict(dict)
    for (i, (key, attr, _)) in enumerate(components):
        if isinstance(attr, str):
            kwargs[key][attr] = world_values[i]
        else:
            while attr > len(args[key]) - 1:
                args[key].append(None)
            args[key][attr] = world_values[i]
    result = []
    for key in default_order(components):
        (klass, ar, kw, *rest) = classes[key]
        if len(rest) == 0:
            klass_gen = klass
        elif len(rest) == 1:
            klass_gen = rest[0]
        else:
            raise ValueError('Tuples in world_axis_object_classes should have length 3 or 4')
        result.append(klass_gen(*args[key], *ar, **kwargs[key], **kw))
    return result

class HighLevelWCSMixin(BaseHighLevelWCS):
    """
    Mix-in class that automatically provides the high-level WCS API for the
    low-level WCS object given by the `~HighLevelWCSMixin.low_level_wcs`
    property.
    """

    @property
    def low_level_wcs(self):
        if False:
            i = 10
            return i + 15
        return self

    def world_to_pixel(self, *world_objects):
        if False:
            return 10
        world_values = high_level_objects_to_values(*world_objects, low_level_wcs=self.low_level_wcs)
        pixel_values = self.low_level_wcs.world_to_pixel_values(*world_values)
        return pixel_values

    def pixel_to_world(self, *pixel_arrays):
        if False:
            while True:
                i = 10
        world_values = self.low_level_wcs.pixel_to_world_values(*pixel_arrays)
        if self.low_level_wcs.world_n_dim == 1:
            world_values = (world_values,)
        pixel_values = values_to_high_level_objects(*world_values, low_level_wcs=self.low_level_wcs)
        if len(pixel_values) == 1:
            return pixel_values[0]
        else:
            return pixel_values