import numpy as np
from astropy.utils.data_info import ParentDtypeInfo

class NdarrayMixinInfo(ParentDtypeInfo):
    _represent_as_dict_primary_data = 'data'

    def _represent_as_dict(self):
        if False:
            i = 10
            return i + 15
        'Represent Column as a dict that can be serialized.'
        col = self._parent
        out = {'data': col.view(np.ndarray)}
        return out

    def _construct_from_dict(self, map):
        if False:
            for i in range(10):
                print('nop')
        'Construct Column from ``map``.'
        data = map.pop('data')
        out = self._parent_cls(data, **map)
        return out

class NdarrayMixin(np.ndarray):
    """
    Mixin column class to allow storage of arbitrary numpy
    ndarrays within a Table.  This is a subclass of numpy.ndarray
    and has the same initialization options as ``np.array()``.
    """
    info = NdarrayMixinInfo()

    def __new__(cls, obj, *args, **kwargs):
        if False:
            while True:
                i = 10
        self = np.array(obj, *args, **kwargs).view(cls)
        if 'info' in getattr(obj, '__dict__', ()):
            self.info = obj.info
        return self

    def __array_finalize__(self, obj):
        if False:
            for i in range(10):
                print('nop')
        if obj is None:
            return
        if callable(super().__array_finalize__):
            super().__array_finalize__(obj)
        if 'info' in getattr(obj, '__dict__', ()):
            self.info = obj.info

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        object_state = list(super().__reduce__())
        object_state[2] = (object_state[2], self.__dict__)
        return tuple(object_state)

    def __setstate__(self, state):
        if False:
            i = 10
            return i + 15
        (nd_state, own_state) = state
        super().__setstate__(nd_state)
        self.__dict__.update(own_state)