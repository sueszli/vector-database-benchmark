__all__ = ['bool_types', 'int_types', 'float_types', 'complex_types', 'continuous_types', 'discrete_types', 'typefilter', 'isgenerator']
bool_types = {'int8'}
int_types = {'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'}
float_types = {'float32', 'float64'}
complex_types = {'complex64', 'complex128'}
continuous_types = float_types | complex_types
discrete_types = bool_types | int_types
string_types = str

def typefilter(vars, types):
    if False:
        while True:
            i = 10
    return [v for v in vars if v.dtype in types]

def isgenerator(obj):
    if False:
        return 10
    return hasattr(obj, '__next__')