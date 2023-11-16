from functools import reduce
from paddle.framework import core
from paddle.framework.io import Variable
dtype_to_size = {core.VarDesc.VarType.FP16: 2, core.VarDesc.VarType.FP32: 4, core.VarDesc.VarType.FP64: 8, core.VarDesc.VarType.INT16: 2, core.VarDesc.VarType.INT32: 4, core.VarDesc.VarType.INT64: 8, core.VarDesc.VarType.BOOL: 1, core.VarDesc.VarType.UINT8: 1}

class VarBlock:

    def __init__(self, varname, offset, size):
        if False:
            for i in range(10):
                print('nop')
        self.varname = varname
        self.offset = offset
        self.size = size

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '%s:%d:%d' % (self.varname, self.offset, self.size)

def create_var_struct(var):
    if False:
        return 10
    if var.type == core.VarDesc.VarType.SELECTED_ROWS:
        lod_level = None
    elif var.type == core.VarDesc.VarType.LOD_TENSOR:
        lod_level = var.lod_level
    else:
        raise ValueError('can only support SELECTED_ROWS/LOD_TENSOR now')
    return VarStruct(var.name, var.shape, var.dtype, var.type, lod_level, var.persistable)

class VarStruct:
    """
    record part properties of a Variable in python.
    """

    def __init__(self, name, shape, dtype, type, lod_level, persistable):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.type = type
        self.lod_level = lod_level
        self.persistable = persistable
        self.m_size = 1
        self.m_size = reduce(lambda x, y: x * y, shape, 1)
        self.m_size *= dtype_to_size[dtype]

    def __str__(self):
        if False:
            print('Hello World!')
        return 'N: {}, S: {}, D: {}, T: {}, LL: {}, P: {}, M: {}'.format(self.name, self.shape, self.dtype, self.type, self.lod_level, self.persistable, self.m_size)

class VarDistributed:
    """
    a class to record the var distributed on parameter servers.
    the class will record the relationship between origin var and slice var.
    the slice var's properties, such as type/shape/offset/endpoint.
    """

    def __init__(self, origin_var, slice_var, is_slice=None, block_id=None, offset=None, vtype=None, endpoint=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            origin_var(Variable|VarStruct): origin var properties\n            slice_var(Variable|VarStruct): slice var properties\n            is_slice(bool|None): slice or not, slice_var=True/False and its block size > 8192 are the judgement standard.\n            block_id(int|None): the number about the slice var.\n            offset(int|None): if the slice var is sliced, offset is the numel before the var.\n            vtype(str|None): a tag, such as Optimizer/Param/RemoteProfetch.\n            endpoint(str|None): which parameter the slice var on, such as "127.0.0.1:1001"\n        '
        if isinstance(origin_var, Variable):
            self.origin = create_var_struct(origin_var)
        else:
            self.origin = origin_var
        if isinstance(slice_var, Variable):
            self.slice = create_var_struct(slice_var)
        else:
            self.slice = slice_var
        if self.equal(self.origin, self.slice):
            self.is_slice = False
            self.block_id = 0
            self.offset = 0
        else:
            self.is_slice = True
            self.block_id = 0
            self.offset = 0
        if is_slice is not None:
            self.is_slice = is_slice
        if block_id is not None:
            self.block_id = block_id
        if offset is not None:
            self.offset = offset
        self.vtype = vtype
        self.endpoint = endpoint

    @staticmethod
    def equal(var1, var2):
        if False:
            return 10
        '\n        the two var is equal or not.\n        Returns:\n            bool: equal will return True else False\n        '
        assert isinstance(var1, VarStruct) and isinstance(var2, VarStruct)
        return var1.name == var2.name and var1.type == var2.type and (var1.shape == var2.shape) and (var1.dtype == var2.dtype) and (var1.lod_level == var2.lod_level) and (var1.persistable == var2.persistable)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        origin_var_str = '{name} : base.{type}.shape{shape}.astype({dtype})'.format(name=self.origin.name, type=self.origin.type, shape=self.origin.shape, dtype=self.origin.dtype)
        slice_var_str = '{name} : base.{type}.shape{shape}.astype({dtype}).slice({is_slice}).block({block_id}).offset({offset})'.format(name=self.slice.name, type=self.slice.type, shape=self.slice.shape, dtype=self.slice.dtype, is_slice=self.is_slice, block_id=self.block_id, offset=self.offset)
        return 'var owned: {}, origin var: ( {} ), slice var: ( {} ), endpoint: {} '.format(self.vtype, origin_var_str, slice_var_str, self.endpoint)

class VarsDistributed:
    """
    a gather about VarDistributed with many methods to find distributed vars.
    through the class, we can get overview about the distributed parameters on parameter servers.
    this class may centralized and convenient for developer to manage and get variable's distribute.
    other module can also use this to find variables such io.py.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.distributed_vars = []

    def add_distributed_var(self, origin_var, slice_var, is_slice=None, block_id=None, offset=None, vtype=None, endpoint=None):
        if False:
            while True:
                i = 10
        '\n        add distributed var in this.\n\n        Args:\n            origin_var(Variable|VarStruct): origin var properties\n            slice_var(Variable|VarStruct): slice var properties\n            is_slice(bool|None): slice or not, slice_var=True/False and its block size > 8192 are the judgement standard.\n            block_id(int|None): the number about the slice var.\n            offset(int|None): if the slice var is sliced, offset is the numel before the var.\n            vtype(str|None): a tag, such as Optimizer/Param/RemoteProfetch.\n            endpoint(str|None): which parameter the slice var on, such as "127.0.0.1:1001"\n        Returns:\n            None\n        '
        self.distributed_vars.append(VarDistributed(origin_var, slice_var, is_slice, block_id, offset, vtype, endpoint))