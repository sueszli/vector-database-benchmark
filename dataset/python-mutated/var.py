from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.types.symbolic import is_symbolic, any_symbolic

class Var(object):
    """
    Var represents the outputs of an Operation. Most Vars are derived from an
    Operation (including const), and all Vars must have `sym_type`.

    Example Usage:

    from coremltools.converters.mil.mil import Builder as mb
    from coremltools.converters.mil.mil import Function
    from coremltools.converters.mil.mil import types

    func_inputs = {"a": mb.placeholder(shape=(1,2)),
                   "b": mb.placeholder(shape=(1,2)) }
    with Function(func_inputs) as ssa_func:
        a, b = ssa_func.inputs["a"], ssa_func.inputs["b"]
        res = mb.add(x=a, y=b) # res is Var
        assert types.is_tensor(res.sym_type)
        assert res.rank == 2
        assert res.dtype == types.float # since a, b are by default float

        # value is not available at compile time in this  case. If
        # materializable, res.val would be a numpy / primitive value
        assert res.val is None


    Comment: Except InternalVar and Vars created in while_loop and by
    placeholder, all Var should only be constructed by Operation to represent
    outputs.

    Comment: Var hides the details of sym_type vs sym_val vs materialized
    value, which was represented by 2 objects prior to refactoring.


    # Properties:

    name: (str)
        name in MIL proto NamedValueType. Name is assigned by the parent
        Operation.

    sym_type [_sym_type]: (builtin type class)
        All Var must have a (possibly symbolic) type, usually derived from
        type inference of upstream ops or from default values in _Input.

    sym_val [_sym_val]: (builtin type instance)
        Possibly symbolic value.

    val [_sym_val]: (np.ndarray or python primitive scalar)
        Numpy (scalar / tensor) value. `val` is not None iff `sym_val` is
        not None and does not contain symbols.  Read-only.

    op [_op]: (Operation)
        The Operation this Var is derived from. May not be None except
        for InternalVar. Read-only.

    op_output_idx: (int)
        Idx of the output from Operation corresponding to _Input.  May be
        None.

    child_ops [_child_ops]: list[Operation]
        Ops that take this Var as an input.
    """
    __slots__ = ['name', '_sym_type', '_sym_val', '_op', 'op_output_idx', '_child_ops', 'consuming_blocks']

    def __init__(self, name, sym_type, sym_val=None, op=None, op_output_idx=None):
        if False:
            while True:
                i = 10
        '\n        sym_type (builtin type)\n        sym_val (builtin value)\n        op (Operation)\n        op_output_idx (int)\n        '
        self.name = name
        self._sym_type = sym_type
        self._sym_val = sym_val
        self._op = op
        self.op_output_idx = op_output_idx
        self._child_ops = list()
        self.consuming_blocks = list()

    @property
    def sym_type(self):
        if False:
            print('Hello World!')
        return self._sym_type

    @property
    def shape(self):
        if False:
            return 10
        if types.is_tensor(self._sym_type):
            return self._sym_type.get_shape()
        return tuple()

    @property
    def rank(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.shape)

    @property
    def dtype(self):
        if False:
            print('Hello World!')
        if types.is_tensor(self._sym_type):
            return self._sym_type.get_primitive()
        return self._sym_type

    @property
    def sym_val(self):
        if False:
            return 10
        if self._sym_val is None:
            return None
        return self._sym_val.val

    @property
    def val(self):
        if False:
            while True:
                i = 10
        if self._sym_val is None or any_symbolic(self._sym_val.val):
            return None
        return self._sym_val.val

    @property
    def op(self):
        if False:
            print('Hello World!')
        return self._op

    @property
    def child_ops(self):
        if False:
            print('Hello World!')
        return self._child_ops

    def add_child_op(self, new_op):
        if False:
            i = 10
            return i + 15
        self._child_ops.append(new_op)

    def remove_child_op(self, target_op, no_check=False):
        if False:
            print('Hello World!')
        if target_op not in self._child_ops:
            if no_check:
                return
            msg = 'Op {} does not takes Var {} as input'
            raise ValueError(msg.format(target_op.name, self.name))
        self._child_ops.remove(target_op)

    def shape_str(self):
        if False:
            for i in range(10):
                print('nop')
        annotation = ''
        if self.val is not None:
            annotation = '*'
        elif self.sym_val is not None:
            annotation = '^'
        shape_str = str(self.shape)[:-1]
        if self.rank > 1:
            shape_str += ', '
        shape_str += types.builtin_to_string(self.dtype) + ')' + annotation
        return shape_str

    def set_name(self, name):
        if False:
            print('Hello World!')
        self.name = name

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '%' + self.name + ': ' + self.shape_str()

class ListVar(Var):
    __slots__ = ['_elem_type', 'init_length', 'dynamic_length']

    def __init__(self, name, elem_type=None, init_length=None, dynamic_length=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        elem_type (builtin.tensor)\n\n        init_length (int): initial length\n\n        dynamic_length (bool): True to allow list to grow. False uses\n        init_length as the fixed size (init_length is runtime length).\n        '
        super(ListVar, self).__init__(name=name, sym_type=types.list(elem_type, init_length, dynamic_length), sym_val=None, **kwargs)
        self._elem_type = elem_type
        self.init_length = init_length
        self.dynamic_length = dynamic_length

    @property
    def shape(self):
        if False:
            return 10
        raise ValueError("shape not applicable to ListVar '{}'.".format(self.name))

    @property
    def rank(self):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError("rank not applicable to ListVar '{}'".format(self.name))

    @property
    def dtype(self):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError("dtype not applicable to ListVar '{}'".format(self.name))

    @property
    def elem_type(self):
        if False:
            i = 10
            return i + 15
        return self._elem_type

    @property
    def elem_shape(self):
        if False:
            return 10
        if self._elem_type == types.unknown:
            return None
        return self._elem_type.get_shape()

    def shape_str(self):
        if False:
            i = 10
            return i + 15
        length = '?'
        if not self.dynamic_length:
            length = str(self.init_length)
        if self._elem_type == types.unknown:
            return 'List[{}, unknown]'.format(length)
        elem_shape = self._elem_type.get_shape()
        elem_dtype = self._elem_type.get_primitive()
        shape_str = str(elem_shape)[:-1]
        if len(elem_shape) > 1:
            shape_str += ', '
        shape_str += types.builtin_to_string(elem_dtype) + ')'
        return 'List[{}, {}]'.format(length, shape_str)

class InternalVar(Var):
    """
    Internal Var (with '__' prefix and won't appear in SSA) will ALWAYS have
    `sym_val == builtin.unknown`. InternalVar are constructed by builder only.

    Comment: Internal Var can be used to represent diverse types such as enum
    type `DataType.FLOAT32`.
    """

    def __init__(self, val, name=None):
        if False:
            print('Hello World!')
        super(InternalVar, self).__init__(name=name, sym_type=types.unknown, sym_val=types.unknown(val))