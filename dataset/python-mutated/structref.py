"""Utilities for defining a mutable struct.

A mutable struct is passed by reference;
hence, structref (a reference to a struct).

"""
from numba import njit
from numba.core import types, imputils, cgutils
from numba.core.datamodel import default_manager, models
from numba.core.extending import infer_getattr, lower_getattr_generic, lower_setattr_generic, box, unbox, NativeValue, intrinsic, overload
from numba.core.typing.templates import AttributeTemplate

class _Utils:
    """Internal builder-code utils for structref definitions.
    """

    def __init__(self, context, builder, struct_type):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        context :\n            a numba target context\n        builder :\n            a llvmlite IRBuilder\n        struct_type : numba.core.types.StructRef\n        '
        self.context = context
        self.builder = builder
        self.struct_type = struct_type

    def new_struct_ref(self, mi):
        if False:
            print('Hello World!')
        'Encapsulate the MemInfo from a `StructRefPayload` in a `StructRef`\n        '
        context = self.context
        builder = self.builder
        struct_type = self.struct_type
        st = cgutils.create_struct_proxy(struct_type)(context, builder)
        st.meminfo = mi
        return st

    def get_struct_ref(self, val):
        if False:
            for i in range(10):
                print('nop')
        'Return a helper for accessing a StructRefType\n        '
        context = self.context
        builder = self.builder
        struct_type = self.struct_type
        return cgutils.create_struct_proxy(struct_type)(context, builder, value=val)

    def get_data_pointer(self, val):
        if False:
            while True:
                i = 10
        'Get the data pointer to the payload from a `StructRefType`.\n        '
        context = self.context
        builder = self.builder
        struct_type = self.struct_type
        structval = self.get_struct_ref(val)
        meminfo = structval.meminfo
        data_ptr = context.nrt.meminfo_data(builder, meminfo)
        valtype = struct_type.get_data_type()
        model = context.data_model_manager[valtype]
        alloc_type = model.get_value_type()
        data_ptr = builder.bitcast(data_ptr, alloc_type.as_pointer())
        return data_ptr

    def get_data_struct(self, val):
        if False:
            while True:
                i = 10
        'Get a getter/setter helper for accessing a `StructRefPayload`\n        '
        context = self.context
        builder = self.builder
        struct_type = self.struct_type
        data_ptr = self.get_data_pointer(val)
        valtype = struct_type.get_data_type()
        dataval = cgutils.create_struct_proxy(valtype)(context, builder, ref=data_ptr)
        return dataval

def define_attributes(struct_typeclass):
    if False:
        while True:
            i = 10
    'Define attributes on `struct_typeclass`.\n\n    Defines both setters and getters in jit-code.\n\n    This is called directly in `register()`.\n    '

    @infer_getattr
    class StructAttribute(AttributeTemplate):
        key = struct_typeclass

        def generic_resolve(self, typ, attr):
            if False:
                for i in range(10):
                    print('nop')
            if attr in typ.field_dict:
                attrty = typ.field_dict[attr]
                return attrty

    @lower_getattr_generic(struct_typeclass)
    def struct_getattr_impl(context, builder, typ, val, attr):
        if False:
            i = 10
            return i + 15
        utils = _Utils(context, builder, typ)
        dataval = utils.get_data_struct(val)
        ret = getattr(dataval, attr)
        fieldtype = typ.field_dict[attr]
        return imputils.impl_ret_borrowed(context, builder, fieldtype, ret)

    @lower_setattr_generic(struct_typeclass)
    def struct_setattr_impl(context, builder, sig, args, attr):
        if False:
            while True:
                i = 10
        [inst_type, val_type] = sig.args
        [instance, val] = args
        utils = _Utils(context, builder, inst_type)
        dataval = utils.get_data_struct(instance)
        field_type = inst_type.field_dict[attr]
        casted = context.cast(builder, val, val_type, field_type)
        old_value = getattr(dataval, attr)
        context.nrt.incref(builder, val_type, casted)
        context.nrt.decref(builder, val_type, old_value)
        setattr(dataval, attr, casted)

def define_boxing(struct_type, obj_class):
    if False:
        for i in range(10):
            print('nop')
    'Define the boxing & unboxing logic for `struct_type` to `obj_class`.\n\n    Defines both boxing and unboxing.\n\n    - boxing turns an instance of `struct_type` into a PyObject of `obj_class`\n    - unboxing turns an instance of `obj_class` into an instance of\n      `struct_type` in jit-code.\n\n\n    Use this directly instead of `define_proxy()` when the user does not\n    want any constructor to be defined.\n    '
    if struct_type is types.StructRef:
        raise ValueError(f'cannot register {types.StructRef}')
    obj_ctor = obj_class._numba_box_

    @box(struct_type)
    def box_struct_ref(typ, val, c):
        if False:
            return 10
        '\n        Convert a raw pointer to a Python int.\n        '
        utils = _Utils(c.context, c.builder, typ)
        struct_ref = utils.get_struct_ref(val)
        meminfo = struct_ref.meminfo
        mip_type = types.MemInfoPointer(types.voidptr)
        boxed_meminfo = c.box(mip_type, meminfo)
        ctor_pyfunc = c.pyapi.unserialize(c.pyapi.serialize_object(obj_ctor))
        ty_pyobj = c.pyapi.unserialize(c.pyapi.serialize_object(typ))
        res = c.pyapi.call_function_objargs(ctor_pyfunc, [ty_pyobj, boxed_meminfo])
        c.pyapi.decref(ctor_pyfunc)
        c.pyapi.decref(ty_pyobj)
        c.pyapi.decref(boxed_meminfo)
        return res

    @unbox(struct_type)
    def unbox_struct_ref(typ, obj, c):
        if False:
            return 10
        mi_obj = c.pyapi.object_getattr_string(obj, '_meminfo')
        mip_type = types.MemInfoPointer(types.voidptr)
        mi = c.unbox(mip_type, mi_obj).value
        utils = _Utils(c.context, c.builder, typ)
        struct_ref = utils.new_struct_ref(mi)
        out = struct_ref._getvalue()
        c.pyapi.decref(mi_obj)
        return NativeValue(out)

def define_constructor(py_class, struct_typeclass, fields):
    if False:
        i = 10
        return i + 15
    'Define the jit-code constructor for `struct_typeclass` using the\n    Python type `py_class` and the required `fields`.\n\n    Use this instead of `define_proxy()` if the user does not want boxing\n    logic defined.\n    '
    params = ', '.join(fields)
    indent = ' ' * 8
    init_fields_buf = []
    for k in fields:
        init_fields_buf.append(f'st.{k} = {k}')
    init_fields = f'\n{indent}'.join(init_fields_buf)
    source = f'\ndef ctor({params}):\n    struct_type = struct_typeclass(list(zip({list(fields)}, [{params}])))\n    def impl({params}):\n        st = new(struct_type)\n        {init_fields}\n        return st\n    return impl\n'
    glbs = dict(struct_typeclass=struct_typeclass, new=new)
    exec(source, glbs)
    ctor = glbs['ctor']
    overload(py_class)(ctor)

def define_proxy(py_class, struct_typeclass, fields):
    if False:
        return 10
    'Defines a PyObject proxy for a structref.\n\n    This makes `py_class` a valid constructor for creating a instance of\n    `struct_typeclass` that contains the members as defined by `fields`.\n\n    Parameters\n    ----------\n    py_class : type\n        The Python class for constructing an instance of `struct_typeclass`.\n    struct_typeclass : numba.core.types.Type\n        The structref type class to bind to.\n    fields : Sequence[str]\n        A sequence of field names.\n\n    Returns\n    -------\n    None\n    '
    define_constructor(py_class, struct_typeclass, fields)
    define_boxing(struct_typeclass, py_class)

def register(struct_type):
    if False:
        while True:
            i = 10
    'Register a `numba.core.types.StructRef` for use in jit-code.\n\n    This defines the data-model for lowering an instance of `struct_type`.\n    This defines attributes accessor and mutator for an instance of\n    `struct_type`.\n\n    Parameters\n    ----------\n    struct_type : type\n        A subclass of `numba.core.types.StructRef`.\n\n    Returns\n    -------\n    struct_type : type\n        Returns the input argument so this can act like a decorator.\n\n    Examples\n    --------\n\n    .. code-block::\n\n        class MyStruct(numba.core.types.StructRef):\n            ...  # the simplest subclass can be empty\n\n        numba.experimental.structref.register(MyStruct)\n\n    '
    if struct_type is types.StructRef:
        raise ValueError(f'cannot register {types.StructRef}')
    default_manager.register(struct_type, models.StructRefModel)
    define_attributes(struct_type)
    return struct_type

@intrinsic
def new(typingctx, struct_type):
    if False:
        while True:
            i = 10
    'new(struct_type)\n\n    A jit-code only intrinsic. Used to allocate an **empty** mutable struct.\n    The fields are zero-initialized and must be set manually after calling\n    the function.\n\n    Example:\n\n        instance = new(MyStruct)\n        instance.field = field_value\n    '
    from numba.experimental.jitclass.base import imp_dtor
    inst_type = struct_type.instance_type

    def codegen(context, builder, signature, args):
        if False:
            print('Hello World!')
        model = context.data_model_manager[inst_type.get_data_type()]
        alloc_type = model.get_value_type()
        alloc_size = context.get_abi_sizeof(alloc_type)
        meminfo = context.nrt.meminfo_alloc_dtor(builder, context.get_constant(types.uintp, alloc_size), imp_dtor(context, builder.module, inst_type))
        data_pointer = context.nrt.meminfo_data(builder, meminfo)
        data_pointer = builder.bitcast(data_pointer, alloc_type.as_pointer())
        builder.store(cgutils.get_null_value(alloc_type), data_pointer)
        inst_struct = context.make_helper(builder, inst_type)
        inst_struct.meminfo = meminfo
        return inst_struct._getvalue()
    sig = inst_type(struct_type)
    return (sig, codegen)

class StructRefProxy:
    """A PyObject proxy to the Numba allocated structref data structure.

    Notes
    -----

    * Subclasses should not define ``__init__``.
    * Subclasses can override ``__new__``.
    """
    __slots__ = ('_type', '_meminfo')

    @classmethod
    def _numba_box_(cls, ty, mi):
        if False:
            while True:
                i = 10
        'Called by boxing logic, the conversion of Numba internal\n        representation into a PyObject.\n\n        Parameters\n        ----------\n        ty :\n            a Numba type instance.\n        mi :\n            a wrapped MemInfoPointer.\n\n        Returns\n        -------\n        instance :\n             a StructRefProxy instance.\n        '
        instance = super().__new__(cls)
        instance._type = ty
        instance._meminfo = mi
        return instance

    def __new__(cls, *args):
        if False:
            print('Hello World!')
        'Construct a new instance of the structref.\n\n        This takes positional-arguments only due to limitation of the compiler.\n        The arguments are mapped to ``cls(*args)`` in jit-code.\n        '
        try:
            ctor = cls.__numba_ctor
        except AttributeError:

            @njit
            def ctor(*args):
                if False:
                    return 10
                return cls(*args)
            cls.__numba_ctor = ctor
        return ctor(*args)

    @property
    def _numba_type_(self):
        if False:
            while True:
                i = 10
        'Returns the Numba type instance for this structref instance.\n\n        Subclasses should NOT override.\n        '
        return self._type