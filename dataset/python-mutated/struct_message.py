import struct

class Structure:
    """
    Utility around Python `struct` module (https://docs.python.org/3.6/library/struct.html)
     that allows to access and modify `_fields` like an ordinary object attributes
     and read/write their values from/into the buffer in C struct like format.
    Similar approach of declaring _fields_ with corresponding C types can be found in
    Python `ctypes` module (https://docs.python.org/3/library/ctypes.html).
    """
    _fields = tuple()

    def __init__(self, *values):
        if False:
            for i in range(10):
                print('nop')
        self.setup_struct()
        self.set_values(*values)

    @classmethod
    def setup_struct(cls):
        if False:
            i = 10
            return i + 15
        if '_struct_desc' not in cls.__dict__:
            cls._struct_desc = '@' + ''.join((field_type for (_, field_type) in cls._fields))
            cls._struct = struct.Struct(cls._struct_desc)

    def __getstate__(self):
        if False:
            print('Hello World!')
        return self.__dict__.copy()

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.__dict__.update(state)
        self.setup_struct()

    def set_values(self, *values):
        if False:
            return 10
        for ((field_name, _), value) in zip(self._fields, values):
            setattr(self, field_name, value)

    def get_values(self):
        if False:
            while True:
                i = 10
        return tuple((getattr(self, field_name) for (field_name, _) in self._fields))

    def pack_into(self, buf, offset):
        if False:
            i = 10
            return i + 15
        try:
            values = self.get_values()
            return self._struct.pack_into(buf, offset, *values)
        except struct.error as e:
            raise RuntimeError('Failed to serialize object as C-like structure. Tried to populate following fields: `{}` with respective values: `{}` '.format(self._fields, self.get_values())) from e

    def unpack_from(self, buf, offset):
        if False:
            print('Hello World!')
        values = self._struct.unpack_from(buf, offset)
        self.set_values(*values)
        return self

    def get_size(self):
        if False:
            while True:
                i = 10
        return self._struct.size