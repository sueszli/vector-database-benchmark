"""
C helper for Miasm:
* raw C to Miasm expression
* Miasm expression to raw C
* Miasm expression to C type
"""
from builtins import zip
from builtins import int as int_types
import warnings
from pycparser import c_parser, c_ast
from functools import total_ordering
from miasm.core.utils import cmp_elts
from miasm.expression.expression_reduce import ExprReducer
from miasm.expression.expression import ExprInt, ExprId, ExprOp, ExprMem
from miasm.arch.x86.arch import is_op_segm
from miasm.core.ctypesmngr import CTypeUnion, CTypeStruct, CTypeId, CTypePtr, CTypeArray, CTypeOp, CTypeSizeof, CTypeEnum, CTypeFunc, CTypeEllipsis
PADDING_TYPE_NAME = '___padding___'

def missing_definition(objtype):
    if False:
        for i in range(10):
            print('nop')
    warnings.warn('Null size type: Missing definition? %r' % objtype)
'\nDisplay C type\nsource: "The C Programming Language - 2nd Edition - Ritchie Kernighan.pdf"\np. 124\n'

def objc_to_str(objc, result=None):
    if False:
        return 10
    if result is None:
        result = ''
    while True:
        if isinstance(objc, ObjCArray):
            result += '[%d]' % objc.elems
            objc = objc.objtype
        elif isinstance(objc, ObjCPtr):
            if not result and isinstance(objc.objtype, ObjCFunc):
                result = objc.objtype.name
            if isinstance(objc.objtype, (ObjCPtr, ObjCDecl, ObjCStruct, ObjCUnion)):
                result = '*%s' % result
            else:
                result = '(*%s)' % result
            objc = objc.objtype
        elif isinstance(objc, (ObjCDecl, ObjCStruct, ObjCUnion)):
            if result:
                result = '%s %s' % (objc, result)
            else:
                result = str(objc)
            break
        elif isinstance(objc, ObjCFunc):
            args_str = []
            for (name, arg) in objc.args:
                args_str.append(objc_to_str(arg, name))
            args = ', '.join(args_str)
            result += '(%s)' % args
            objc = objc.type_ret
        elif isinstance(objc, ObjCInt):
            return 'int'
        elif isinstance(objc, ObjCEllipsis):
            return '...'
        else:
            raise TypeError('Unknown c type')
    return result

@total_ordering
class ObjC(object):
    """Generic ObjC"""

    def __init__(self, align, size):
        if False:
            return 10
        self._align = align
        self._size = size

    @property
    def align(self):
        if False:
            return 10
        'Alignment (in bytes) of the C object'
        return self._align

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        'Size (in bytes) of the C object'
        return self._size

    def cmp_base(self, other):
        if False:
            print('Hello World!')
        assert self.__class__ in OBJC_PRIO
        assert other.__class__ in OBJC_PRIO
        if OBJC_PRIO[self.__class__] != OBJC_PRIO[other.__class__]:
            return cmp_elts(OBJC_PRIO[self.__class__], OBJC_PRIO[other.__class__])
        if self.align != other.align:
            return cmp_elts(self.align, other.align)
        return cmp_elts(self.size, other.size)

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((self.__class__, self._align, self._size))

    def __str__(self):
        if False:
            print('Hello World!')
        return objc_to_str(self)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.cmp_base(other) == 0

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self == other

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.cmp_base(other) < 0

@total_ordering
class ObjCDecl(ObjC):
    """C Declaration identified"""

    def __init__(self, name, align, size):
        if False:
            return 10
        super(ObjCDecl, self).__init__(align, size)
        self._name = name
    name = property(lambda self: self._name)

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((super(ObjCDecl, self).__hash__(), self._name))

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s %s>' % (self.__class__.__name__, self.name)

    def __str__(self):
        if False:
            return 10
        return str(self.name)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        ret = self.cmp_base(other)
        if ret:
            return False
        return self.name == other.name

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        ret = self.cmp_base(other)
        if ret:
            if ret < 0:
                return True
            return False
        return self.name < other.name

class ObjCInt(ObjC):
    """C integer"""

    def __init__(self):
        if False:
            while True:
                i = 10
        super(ObjCInt, self).__init__(None, 0)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'int'

@total_ordering
class ObjCPtr(ObjC):
    """C Pointer"""

    def __init__(self, objtype, void_p_align, void_p_size):
        if False:
            while True:
                i = 10
        'Init ObjCPtr\n\n        @objtype: pointer target ObjC\n        @void_p_align: pointer alignment (in bytes)\n        @void_p_size: pointer size (in bytes)\n        '
        super(ObjCPtr, self).__init__(void_p_align, void_p_size)
        self._lock = False
        self.objtype = objtype
        if objtype is None:
            self._lock = False

    def get_objtype(self):
        if False:
            while True:
                i = 10
        assert self._lock is True
        return self._objtype

    def set_objtype(self, objtype):
        if False:
            while True:
                i = 10
        assert self._lock is False
        self._lock = True
        self._objtype = objtype
    objtype = property(get_objtype, set_objtype)

    def __hash__(self):
        if False:
            return 10
        assert self._lock
        return hash((super(ObjCPtr, self).__hash__(), hash(self._objtype)))

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s %r>' % (self.__class__.__name__, self.objtype.__class__)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        ret = self.cmp_base(other)
        if ret:
            return False
        return self.objtype == other.objtype

    def __lt__(self, other):
        if False:
            return 10
        ret = self.cmp_base(other)
        if ret:
            if ret < 0:
                return True
            return False
        return self.objtype < other.objtype

@total_ordering
class ObjCArray(ObjC):
    """C array (test[XX])"""

    def __init__(self, objtype, elems):
        if False:
            print('Hello World!')
        'Init ObjCArray\n\n        @objtype: pointer target ObjC\n        @elems: number of elements in the array\n        '
        super(ObjCArray, self).__init__(objtype.align, elems * objtype.size)
        self._elems = elems
        self._objtype = objtype
    objtype = property(lambda self: self._objtype)
    elems = property(lambda self: self._elems)

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((super(ObjCArray, self).__hash__(), self._elems, hash(self._objtype)))

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%r[%d]>' % (self.objtype, self.elems)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        ret = self.cmp_base(other)
        if ret:
            return False
        if self.objtype != other.objtype:
            return False
        return self.elems == other.elems

    def __lt__(self, other):
        if False:
            return 10
        ret = self.cmp_base(other)
        if ret > 0:
            return False
        if self.objtype > other.objtype:
            return False
        return self.elems < other.elems

@total_ordering
class ObjCStruct(ObjC):
    """C object for structures"""

    def __init__(self, name, align, size, fields):
        if False:
            while True:
                i = 10
        super(ObjCStruct, self).__init__(align, size)
        self._name = name
        self._fields = tuple(fields)
    name = property(lambda self: self._name)
    fields = property(lambda self: self._fields)

    def __hash__(self):
        if False:
            return 10
        return hash((super(ObjCStruct, self).__hash__(), self._name))

    def __repr__(self):
        if False:
            while True:
                i = 10
        out = []
        out.append('Struct %s: (align: %d)' % (self.name, self.align))
        out.append('  off sz  name')
        for (name, objtype, offset, size) in self.fields:
            out.append('  0x%-3x %-3d %-10s %r' % (offset, size, name, objtype.__class__.__name__))
        return '\n'.join(out)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'struct %s' % self.name

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        ret = self.cmp_base(other)
        if ret:
            return False
        return self.name == other.name

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        ret = self.cmp_base(other)
        if ret:
            if ret < 0:
                return True
            return False
        return self.name < other.name

@total_ordering
class ObjCUnion(ObjC):
    """C object for unions"""

    def __init__(self, name, align, size, fields):
        if False:
            for i in range(10):
                print('nop')
        super(ObjCUnion, self).__init__(align, size)
        self._name = name
        self._fields = tuple(fields)
    name = property(lambda self: self._name)
    fields = property(lambda self: self._fields)

    def __hash__(self):
        if False:
            return 10
        return hash((super(ObjCUnion, self).__hash__(), self._name))

    def __repr__(self):
        if False:
            print('Hello World!')
        out = []
        out.append('Union %s: (align: %d)' % (self.name, self.align))
        out.append('  off sz  name')
        for (name, objtype, offset, size) in self.fields:
            out.append('  0x%-3x %-3d %-10s %r' % (offset, size, name, objtype))
        return '\n'.join(out)

    def __str__(self):
        if False:
            print('Hello World!')
        return 'union %s' % self.name

    def __eq__(self, other):
        if False:
            print('Hello World!')
        ret = self.cmp_base(other)
        if ret:
            return False
        return self.name == other.name

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        ret = self.cmp_base(other)
        if ret:
            if ret < 0:
                return True
            return False
        return self.name < other.name

class ObjCEllipsis(ObjC):
    """C integer"""

    def __init__(self):
        if False:
            print('Hello World!')
        super(ObjCEllipsis, self).__init__(None, None)
    align = property(lambda self: self._align)
    size = property(lambda self: self._size)

@total_ordering
class ObjCFunc(ObjC):
    """C object for Functions"""

    def __init__(self, name, abi, type_ret, args, void_p_align, void_p_size):
        if False:
            return 10
        super(ObjCFunc, self).__init__(void_p_align, void_p_size)
        self._name = name
        self._abi = abi
        self._type_ret = type_ret
        self._args = tuple(args)
    args = property(lambda self: self._args)
    type_ret = property(lambda self: self._type_ret)
    abi = property(lambda self: self._abi)
    name = property(lambda self: self._name)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((super(ObjCFunc, self).__hash__(), hash(self._args), self._name))

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<%s %s>' % (self.__class__.__name__, self.name)

    def __str__(self):
        if False:
            while True:
                i = 10
        out = []
        out.append('Function (%s)  %s: (align: %d)' % (self.abi, self.name, self.align))
        out.append('  ret: %s' % str(self.type_ret))
        out.append('  Args:')
        for (name, arg) in self.args:
            out.append('  %s %s' % (name, arg))
        return '\n'.join(out)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        ret = self.cmp_base(other)
        if ret:
            return False
        return self.name == other.name

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        ret = self.cmp_base(other)
        if ret:
            if ret < 0:
                return True
            return False
        return self.name < other.name
OBJC_PRIO = {ObjC: 0, ObjCDecl: 1, ObjCInt: 2, ObjCPtr: 3, ObjCArray: 4, ObjCStruct: 5, ObjCUnion: 6, ObjCEllipsis: 7, ObjCFunc: 8}

def access_simplifier(expr):
    if False:
        print('Hello World!')
    "Expression visitor to simplify a C access represented in Miasm\n\n    @expr: Miasm expression representing the C access\n\n    Example:\n\n    IN: (In c: ['*(&((&((*(ptr_Test)).a))[0]))'])\n    [ExprOp('deref', ExprOp('addr', ExprOp('[]', ExprOp('addr',\n    ExprOp('field', ExprOp('deref', ExprId('ptr_Test', 64)),\n    ExprId('a', 64))), ExprInt(0x0, 64))))]\n\n    OUT: (In c: ['(ptr_Test)->a'])\n    [ExprOp('->', ExprId('ptr_Test', 64), ExprId('a', 64))]\n    "
    if expr.is_op('addr') and expr.args[0].is_op('[]') and (expr.args[0].args[1] == ExprInt(0, 64)):
        return expr.args[0].args[0]
    elif expr.is_op('[]') and expr.args[0].is_op('addr') and (expr.args[1] == ExprInt(0, 64)):
        return expr.args[0].args[0]
    elif expr.is_op('addr') and expr.args[0].is_op('deref'):
        return expr.args[0].args[0]
    elif expr.is_op('deref') and expr.args[0].is_op('addr'):
        return expr.args[0].args[0]
    elif expr.is_op('field') and expr.args[0].is_op('deref'):
        return ExprOp('->', expr.args[0].args[0], expr.args[1])
    return expr

def access_str(expr):
    if False:
        while True:
            i = 10
    "Return the C string of a C access represented in Miasm\n\n    @expr: Miasm expression representing the C access\n\n    In:\n    ExprOp('->', ExprId('ptr_Test', 64), ExprId('a', 64))\n    OUT:\n    '(ptr_Test)->a'\n    "
    if isinstance(expr, ExprId):
        out = str(expr)
    elif isinstance(expr, ExprInt):
        out = str(int(expr))
    elif expr.is_op('addr'):
        out = '&(%s)' % access_str(expr.args[0])
    elif expr.is_op('deref'):
        out = '*(%s)' % access_str(expr.args[0])
    elif expr.is_op('field'):
        out = '(%s).%s' % (access_str(expr.args[0]), access_str(expr.args[1]))
    elif expr.is_op('->'):
        out = '(%s)->%s' % (access_str(expr.args[0]), access_str(expr.args[1]))
    elif expr.is_op('[]'):
        out = '(%s)[%s]' % (access_str(expr.args[0]), access_str(expr.args[1]))
    else:
        raise RuntimeError('unknown op')
    return out

class CGen(object):
    """Generic object to represent a C expression"""
    default_size = 64

    def __init__(self, ctype):
        if False:
            return 10
        self._ctype = ctype

    @property
    def ctype(self):
        if False:
            i = 10
            return i + 15
        'Type (ObjC instance) of the current object'
        return self._ctype

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.__class__)

    def __eq__(self, other):
        if False:
            return 10
        return self.__class__ == other.__class__ and self._ctype == other.ctype

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self.__eq__(other)

    def to_c(self):
        if False:
            i = 10
            return i + 15
        'Generate corresponding C'
        raise NotImplementedError('Virtual')

    def to_expr(self):
        if False:
            return 10
        'Generate Miasm expression representing the C access'
        raise NotImplementedError('Virtual')

class CGenInt(CGen):
    """Int C object"""

    def __init__(self, integer):
        if False:
            print('Hello World!')
        assert isinstance(integer, int_types)
        self._integer = integer
        super(CGenInt, self).__init__(ObjCInt())

    @property
    def integer(self):
        if False:
            for i in range(10):
                print('nop')
        'Value of the object'
        return self._integer

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((super(CGenInt, self).__hash__(), self._integer))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return super(CGenInt, self).__eq__(other) and self._integer == other.integer

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self.__eq__(other)

    def to_c(self):
        if False:
            i = 10
            return i + 15
        'Generate corresponding C'
        return '0x%X' % self.integer

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<%s %s>' % (self.__class__.__name__, self.integer)

    def to_expr(self):
        if False:
            i = 10
            return i + 15
        'Generate Miasm expression representing the C access'
        return ExprInt(self.integer, self.default_size)

class CGenId(CGen):
    """ID of a C object"""

    def __init__(self, ctype, name):
        if False:
            for i in range(10):
                print('nop')
        self._name = name
        assert isinstance(name, str)
        super(CGenId, self).__init__(ctype)

    @property
    def name(self):
        if False:
            while True:
                i = 10
        'Name of the Id'
        return self._name

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((super(CGenId, self).__hash__(), self._name))

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return super(CGenId, self).__eq__(other) and self._name == other.name

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<%s %s>' % (self.__class__.__name__, self.name)

    def to_c(self):
        if False:
            for i in range(10):
                print('nop')
        'Generate corresponding C'
        return '%s' % self.name

    def to_expr(self):
        if False:
            return 10
        'Generate Miasm expression representing the C access'
        return ExprId(self.name, self.default_size)

class CGenField(CGen):
    """
    Field of a C struct/union

    IN:
    - struct (not ptr struct)
    - field name
    OUT:
    - input type of the field => output type
    - X[] => X[]
    - X => X*
    """

    def __init__(self, struct, field, fieldtype, void_p_align, void_p_size):
        if False:
            while True:
                i = 10
        self._struct = struct
        self._field = field
        assert isinstance(field, str)
        if isinstance(fieldtype, ObjCArray):
            ctype = fieldtype
        else:
            ctype = ObjCPtr(fieldtype, void_p_align, void_p_size)
        super(CGenField, self).__init__(ctype)

    @property
    def struct(self):
        if False:
            i = 10
            return i + 15
        'Structure containing the field'
        return self._struct

    @property
    def field(self):
        if False:
            i = 10
            return i + 15
        'Field name'
        return self._field

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((super(CGenField, self).__hash__(), self._struct, self._field))

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return super(CGenField, self).__eq__(other) and self._struct == other.struct and (self._field == other.field)

    def to_c(self):
        if False:
            while True:
                i = 10
        'Generate corresponding C'
        if isinstance(self.ctype, ObjCArray):
            return '(%s).%s' % (self.struct.to_c(), self.field)
        elif isinstance(self.ctype, ObjCPtr):
            return '&((%s).%s)' % (self.struct.to_c(), self.field)
        else:
            raise RuntimeError('Strange case')

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<%s %s %s>' % (self.__class__.__name__, self.struct, self.field)

    def to_expr(self):
        if False:
            return 10
        'Generate Miasm expression representing the C access'
        if isinstance(self.ctype, ObjCArray):
            return ExprOp('field', self.struct.to_expr(), ExprId(self.field, self.default_size))
        elif isinstance(self.ctype, ObjCPtr):
            return ExprOp('addr', ExprOp('field', self.struct.to_expr(), ExprId(self.field, self.default_size)))
        else:
            raise RuntimeError('Strange case')

class CGenArray(CGen):
    """
    C Array

    This object does *not* deref the source, it only do object casting.

    IN:
    - obj
    OUT:
    - X* => X*
    - ..[][] => ..[]
    - X[] => X*
    """

    def __init__(self, base, elems, void_p_align, void_p_size):
        if False:
            while True:
                i = 10
        ctype = base.ctype
        if isinstance(ctype, ObjCPtr):
            pass
        elif isinstance(ctype, ObjCArray) and isinstance(ctype.objtype, ObjCArray):
            ctype = ctype.objtype
        elif isinstance(ctype, ObjCArray):
            ctype = ObjCPtr(ctype.objtype, void_p_align, void_p_size)
        else:
            raise TypeError('Strange case')
        self._base = base
        self._elems = elems
        super(CGenArray, self).__init__(ctype)

    @property
    def base(self):
        if False:
            for i in range(10):
                print('nop')
        'Base object supporting the array'
        return self._base

    @property
    def elems(self):
        if False:
            i = 10
            return i + 15
        'Number of elements in the array'
        return self._elems

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((super(CGenArray, self).__hash__(), self._base, self._elems))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return super(CGenField, self).__eq__(other) and self._base == other.base and (self._elems == other.elems)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<%s %s>' % (self.__class__.__name__, self.base)

    def to_c(self):
        if False:
            i = 10
            return i + 15
        'Generate corresponding C'
        if isinstance(self.ctype, ObjCPtr):
            out_str = '&((%s)[%d])' % (self.base.to_c(), self.elems)
        elif isinstance(self.ctype, ObjCArray):
            out_str = '(%s)[%d]' % (self.base.to_c(), self.elems)
        else:
            raise RuntimeError('Strange case')
        return out_str

    def to_expr(self):
        if False:
            while True:
                i = 10
        'Generate Miasm expression representing the C access'
        if isinstance(self.ctype, ObjCPtr):
            return ExprOp('addr', ExprOp('[]', self.base.to_expr(), ExprInt(self.elems, self.default_size)))
        elif isinstance(self.ctype, ObjCArray):
            return ExprOp('[]', self.base.to_expr(), ExprInt(self.elems, self.default_size))
        else:
            raise RuntimeError('Strange case')

class CGenDeref(CGen):
    """
    C dereference

    IN:
    - ptr
    OUT:
    - X* => X
    """

    def __init__(self, ptr):
        if False:
            return 10
        assert isinstance(ptr.ctype, ObjCPtr)
        self._ptr = ptr
        super(CGenDeref, self).__init__(ptr.ctype.objtype)

    @property
    def ptr(self):
        if False:
            print('Hello World!')
        'Pointer object'
        return self._ptr

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((super(CGenDeref, self).__hash__(), self._ptr))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return super(CGenField, self).__eq__(other) and self._ptr == other.ptr

    def __repr__(self):
        if False:
            return 10
        return '<%s %s>' % (self.__class__.__name__, self.ptr)

    def to_c(self):
        if False:
            for i in range(10):
                print('nop')
        'Generate corresponding C'
        if not isinstance(self.ptr.ctype, ObjCPtr):
            raise RuntimeError()
        return '*(%s)' % self.ptr.to_c()

    def to_expr(self):
        if False:
            print('Hello World!')
        'Generate Miasm expression representing the C access'
        if not isinstance(self.ptr.ctype, ObjCPtr):
            raise RuntimeError()
        return ExprOp('deref', self.ptr.to_expr())

def ast_get_c_access_expr(ast, expr_types, lvl=0):
    if False:
        return 10
    "Transform C ast object into a C Miasm expression\n\n    @ast: parsed pycparser.c_ast object\n    @expr_types: a dictionary linking ID names to their types\n    @lvl: actual recursion level\n\n    Example:\n\n    IN:\n    StructRef: ->\n      ID: ptr_Test\n      ID: a\n\n    OUT:\n    ExprOp('->', ExprId('ptr_Test', 64), ExprId('a', 64))\n    "
    if isinstance(ast, c_ast.Constant):
        obj = ExprInt(int(ast.value), 64)
    elif isinstance(ast, c_ast.StructRef):
        (name, field) = (ast.name, ast.field.name)
        name = ast_get_c_access_expr(name, expr_types)
        if ast.type == '->':
            s_name = name
            s_field = ExprId(field, 64)
            obj = ExprOp('->', s_name, s_field)
        elif ast.type == '.':
            s_name = name
            s_field = ExprId(field, 64)
            obj = ExprOp('field', s_name, s_field)
        else:
            raise RuntimeError('Unknown struct access')
    elif isinstance(ast, c_ast.UnaryOp) and ast.op == '&':
        tmp = ast_get_c_access_expr(ast.expr, expr_types, lvl + 1)
        obj = ExprOp('addr', tmp)
    elif isinstance(ast, c_ast.ArrayRef):
        tmp = ast_get_c_access_expr(ast.name, expr_types, lvl + 1)
        index = ast_get_c_access_expr(ast.subscript, expr_types, lvl + 1)
        obj = ExprOp('[]', tmp, index)
    elif isinstance(ast, c_ast.ID):
        assert ast.name in expr_types
        obj = ExprId(ast.name, 64)
    elif isinstance(ast, c_ast.UnaryOp) and ast.op == '*':
        tmp = ast_get_c_access_expr(ast.expr, expr_types, lvl + 1)
        obj = ExprOp('deref', tmp)
    else:
        raise NotImplementedError('Unknown type')
    return obj

def parse_access(c_access):
    if False:
        for i in range(10):
            print('nop')
    'Parse C access\n\n    @c_access: C access string\n    '
    main = '\n    int main() {\n    %s;\n    }\n    ' % c_access
    parser = c_parser.CParser()
    node = parser.parse(main, filename='<stdin>')
    access = node.ext[-1].body.block_items[0]
    return access

class ExprToAccessC(ExprReducer):
    """
    Generate the C access object(s) for a given native Miasm expression
    Example:
    IN:
    @32[ptr_Test]
    OUT:
    [<CGenDeref <CGenArray <CGenField <CGenDeref <CGenId ptr_Test>> a>>>]

    An expression may be represented by multiple accessor (due to unions).
    """

    def __init__(self, expr_types, types_mngr, enforce_strict_access=True):
        if False:
            for i in range(10):
                print('nop')
        'Init GenCAccess\n\n        @expr_types: a dictionary linking ID names to their types\n        @types_mngr: types manager\n        @enforce_strict_access: If false, generate access even on expression\n        pointing to a middle of an object. If true, raise exception if such a\n        pointer is encountered\n        '
        self.expr_types = expr_types
        self.types_mngr = types_mngr
        self.enforce_strict_access = enforce_strict_access

    def updt_expr_types(self, expr_types):
        if False:
            for i in range(10):
                print('nop')
        'Update expr_types\n        @expr_types: Dictionary associating name to type\n        '
        self.expr_types = expr_types

    def cgen_access(self, cgenobj, base_type, offset, deref, lvl=0):
        if False:
            for i in range(10):
                print('nop')
        'Return the access(es) which lead to the element at @offset of an\n        object of type @base_type\n\n        In case of no @deref, stops recursion as soon as we reached the base of\n        an object.\n        In other cases, we need to go down to the final dereferenced object\n\n        @cgenobj: current object access\n        @base_type: type of main object\n        @offset: offset (in bytes) of the target sub object\n        @deref: get type for a pointer or a deref\n        @lvl: actual recursion level\n\n\n        IN:\n        - base_type: struct Toto{\n            int a\n            int b\n          }\n        - base_name: var\n        - 4\n        OUT:\n        - CGenField(var, b)\n\n\n\n        IN:\n        - base_type: int a\n        - 0\n        OUT:\n        - CGenAddr(a)\n\n        IN:\n        - base_type: X = int* a\n        - 0\n        OUT:\n        - CGenAddr(X)\n\n        IN:\n        - X = int* a\n        - 8\n        OUT:\n        - ASSERT\n\n\n        IN:\n        - struct toto{\n            int a\n            int b[10]\n          }\n        - 8\n        OUT:\n        - CGenArray(CGenField(toto, b), 1)\n        '
        if base_type.size == 0:
            missing_definition(base_type)
            return set()
        void_type = self.types_mngr.void_ptr
        if isinstance(base_type, ObjCStruct):
            if not 0 <= offset < base_type.size:
                return set()
            if offset == 0 and (not deref):
                return set([cgenobj])
            for (fieldname, subtype, field_offset, size) in base_type.fields:
                if not field_offset <= offset < field_offset + size:
                    continue
                fieldptr = CGenField(CGenDeref(cgenobj), fieldname, subtype, void_type.align, void_type.size)
                new_type = self.cgen_access(fieldptr, subtype, offset - field_offset, deref, lvl + 1)
                break
            else:
                return set()
        elif isinstance(base_type, ObjCArray):
            if base_type.objtype.size == 0:
                missing_definition(base_type.objtype)
                return set()
            element_num = offset // base_type.objtype.size
            field_offset = offset % base_type.objtype.size
            if element_num >= base_type.elems:
                return set()
            if offset == 0 and (not deref):
                return set([cgenobj])
            curobj = CGenArray(cgenobj, element_num, void_type.align, void_type.size)
            if field_offset == 0:
                return set([curobj])
            new_type = self.cgen_access(curobj, base_type.objtype, field_offset, deref, lvl + 1)
        elif isinstance(base_type, ObjCDecl):
            if self.enforce_strict_access and offset % base_type.size != 0:
                return set()
            elem_num = offset // base_type.size
            nobj = CGenArray(cgenobj, elem_num, void_type.align, void_type.size)
            new_type = set([nobj])
        elif isinstance(base_type, ObjCUnion):
            if offset == 0 and (not deref):
                return set([cgenobj])
            out = set()
            for (fieldname, objtype, field_offset, size) in base_type.fields:
                if not field_offset <= offset < field_offset + size:
                    continue
                field = CGenField(CGenDeref(cgenobj), fieldname, objtype, void_type.align, void_type.size)
                out.update(self.cgen_access(field, objtype, offset - field_offset, deref, lvl + 1))
            new_type = out
        elif isinstance(base_type, ObjCPtr):
            elem_num = offset // base_type.size
            if self.enforce_strict_access and offset % base_type.size != 0:
                return set()
            nobj = CGenArray(cgenobj, elem_num, void_type.align, void_type.size)
            new_type = set([nobj])
        else:
            raise NotImplementedError('deref type %r' % base_type)
        return new_type

    def reduce_known_expr(self, node, ctxt, **kwargs):
        if False:
            print('Hello World!')
        'Generate access for known expr'
        if node.expr in ctxt:
            objcs = ctxt[node.expr]
            return set((CGenId(objc, str(node.expr)) for objc in objcs))
        return None

    def reduce_int(self, node, **kwargs):
        if False:
            while True:
                i = 10
        'Generate access for ExprInt'
        if not isinstance(node.expr, ExprInt):
            return None
        return set([CGenInt(int(node.expr))])

    def get_solo_type(self, node):
        if False:
            while True:
                i = 10
        'Return the type of the @node if it has only one possible type,\n        different from not None. In other cases, return None.\n        '
        if node.info is None or len(node.info) != 1:
            return None
        return type(list(node.info)[0].ctype)

    def reduce_op(self, node, lvl=0, **kwargs):
        if False:
            print('Hello World!')
        'Generate access for ExprOp'
        if not (node.expr.is_op('+') or is_op_segm(node.expr)) or len(node.args) != 2:
            return None
        type_arg1 = self.get_solo_type(node.args[1])
        if type_arg1 != ObjCInt:
            return None
        (arg0, arg1) = node.args
        if arg0.info is None:
            return None
        void_type = self.types_mngr.void_ptr
        out = set()
        if not arg1.expr.is_int():
            return None
        ptr_offset = int(arg1.expr)
        for info in arg0.info:
            if isinstance(info.ctype, ObjCArray):
                field_type = info.ctype
            elif isinstance(info.ctype, ObjCPtr):
                field_type = info.ctype.objtype
            else:
                continue
            target_type = info.ctype.objtype
            out.update(self.cgen_access(info, field_type, ptr_offset, False, lvl))
        return out

    def reduce_mem(self, node, lvl=0, **kwargs):
        if False:
            i = 10
            return i + 15
        'Generate access for ExprMem:\n        * @NN[ptr<elem>] -> elem  (type)\n        * @64[ptr<ptr<elem>>] -> ptr<elem>\n        * @32[ptr<struct>] -> struct.00\n        '
        if not isinstance(node.expr, ExprMem):
            return None
        if node.ptr.info is None:
            return None
        assert isinstance(node.ptr.info, set)
        void_type = self.types_mngr.void_ptr
        found = set()
        for subcgenobj in node.ptr.info:
            if isinstance(subcgenobj.ctype, ObjCArray):
                nobj = CGenArray(subcgenobj, 0, void_type.align, void_type.size)
                target = nobj.ctype.objtype
                for finalcgenobj in self.cgen_access(nobj, target, 0, True, lvl):
                    assert isinstance(finalcgenobj.ctype, ObjCPtr)
                    if self.enforce_strict_access and finalcgenobj.ctype.objtype.size != node.expr.size // 8:
                        continue
                    found.add(CGenDeref(finalcgenobj))
            elif isinstance(subcgenobj.ctype, ObjCPtr):
                target = subcgenobj.ctype.objtype
                if isinstance(target, (ObjCStruct, ObjCUnion)):
                    for finalcgenobj in self.cgen_access(subcgenobj, target, 0, True, lvl):
                        target = finalcgenobj.ctype.objtype
                        if self.enforce_strict_access and target.size != node.expr.size // 8:
                            continue
                        found.add(CGenDeref(finalcgenobj))
                elif isinstance(target, ObjCArray):
                    if self.enforce_strict_access and subcgenobj.ctype.size != node.expr.size // 8:
                        continue
                    found.update(self.cgen_access(CGenDeref(subcgenobj), target, 0, False, lvl))
                else:
                    if self.enforce_strict_access and target.size != node.expr.size // 8:
                        continue
                    found.add(CGenDeref(subcgenobj))
        if not found:
            return None
        return found
    reduction_rules = [reduce_known_expr, reduce_int, reduce_op, reduce_mem]

    def get_accesses(self, expr, expr_context=None):
        if False:
            i = 10
            return i + 15
        'Generate C access(es) for the native Miasm expression @expr\n        @expr: native Miasm expression\n        @expr_context: a dictionary linking known expressions to their\n        types. An expression is linked to a tuple of types.\n        '
        if expr_context is None:
            expr_context = self.expr_types
        ret = self.reduce(expr, ctxt=expr_context)
        if ret.info is None:
            return set()
        return ret.info

class ExprCToExpr(ExprReducer):
    """Translate a Miasm expression (representing a C access) into a native
    Miasm expression and its C type:

    Example:

    IN: ((ptr_struct -> f_mini) field x)
    OUT: @32[ptr_struct + 0x80], int


    Tricky cases:
    Struct S0 {
        int x;
        int y[0x10];
    }

    Struct S1 {
        int a;
        S0 toto;
    }

    S1* ptr;

    Case 1:
    ptr->toto => ptr + 0x4
    &(ptr->toto) => ptr + 0x4

    Case 2:
    (ptr->toto).x => @32[ptr + 0x4]
    &((ptr->toto).x) => ptr + 0x4

    Case 3:
    (ptr->toto).y => ptr + 0x8
    &((ptr->toto).y) => ptr + 0x8

    Case 4:
    (ptr->toto).y[1] => @32[ptr + 0x8 + 0x4]
    &((ptr->toto).y[1]) => ptr + 0x8 + 0x4

    """

    def __init__(self, expr_types, types_mngr):
        if False:
            for i in range(10):
                print('nop')
        'Init ExprCAccess\n\n        @expr_types: a dictionary linking ID names to their types\n        @types_mngr: types manager\n        '
        self.expr_types = expr_types
        self.types_mngr = types_mngr

    def updt_expr_types(self, expr_types):
        if False:
            return 10
        'Update expr_types\n        @expr_types: Dictionary associating name to type\n        '
        self.expr_types = expr_types
    CST = 'CST'

    def reduce_known_expr(self, node, ctxt, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Reduce known expressions'
        if str(node.expr) in ctxt:
            objc = ctxt[str(node.expr)]
            out = (node.expr, objc)
        elif node.expr.is_id():
            out = (node.expr, None)
        else:
            out = None
        return out

    def reduce_int(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Reduce ExprInt'
        if not isinstance(node.expr, ExprInt):
            return None
        return self.CST

    def reduce_op_memberof(self, node, **kwargs):
        if False:
            while True:
                i = 10
        'Reduce -> operator'
        if not node.expr.is_op('->'):
            return None
        assert len(node.args) == 2
        out = []
        assert isinstance(node.args[1].expr, ExprId)
        field = node.args[1].expr.name
        (src, src_type) = node.args[0].info
        if src_type is None:
            return None
        assert isinstance(src_type, (ObjCPtr, ObjCArray))
        struct_dst = src_type.objtype
        assert isinstance(struct_dst, ObjCStruct)
        found = False
        for (name, objtype, offset, _) in struct_dst.fields:
            if name != field:
                continue
            expr = src + ExprInt(offset, src.size)
            if isinstance(objtype, (ObjCArray, ObjCStruct, ObjCUnion)):
                pass
            else:
                expr = ExprMem(expr, objtype.size * 8)
            assert not found
            found = True
            out = (expr, objtype)
        assert found
        return out

    def reduce_op_field(self, node, **kwargs):
        if False:
            print('Hello World!')
        'Reduce field operator (Struct or Union)'
        if not node.expr.is_op('field'):
            return None
        assert len(node.args) == 2
        out = []
        assert isinstance(node.args[1].expr, ExprId)
        field = node.args[1].expr.name
        (src, src_type) = node.args[0].info
        struct_dst = src_type
        if isinstance(struct_dst, ObjCStruct):
            found = False
            for (name, objtype, offset, _) in struct_dst.fields:
                if name != field:
                    continue
                expr = src + ExprInt(offset, src.size)
                if isinstance(objtype, ObjCArray):
                    pass
                elif isinstance(objtype, (ObjCStruct, ObjCUnion)):
                    pass
                else:
                    expr = ExprMem(expr, objtype.size * 8)
                assert not found
                found = True
                out = (expr, objtype)
        elif isinstance(struct_dst, ObjCUnion):
            found = False
            for (name, objtype, offset, _) in struct_dst.fields:
                if name != field:
                    continue
                expr = src + ExprInt(offset, src.size)
                if isinstance(objtype, ObjCArray):
                    pass
                elif isinstance(objtype, (ObjCStruct, ObjCUnion)):
                    pass
                else:
                    expr = ExprMem(expr, objtype.size * 8)
                assert not found
                found = True
                out = (expr, objtype)
        else:
            raise NotImplementedError('unknown ObjC')
        assert found
        return out

    def reduce_op_array(self, node, **kwargs):
        if False:
            return 10
        'Reduce array operator'
        if not node.expr.is_op('[]'):
            return None
        assert len(node.args) == 2
        out = []
        assert isinstance(node.args[1].expr, ExprInt)
        cst = node.args[1].expr
        (src, src_type) = node.args[0].info
        objtype = src_type.objtype
        expr = src + cst * ExprInt(objtype.size, cst.size)
        if isinstance(src_type, ObjCPtr):
            if isinstance(objtype, ObjCArray):
                final = objtype.objtype
                expr = src + cst * ExprInt(final.size, cst.size)
                objtype = final
                expr = ExprMem(expr, final.size * 8)
                found = True
            else:
                expr = ExprMem(expr, objtype.size * 8)
                found = True
        elif isinstance(src_type, ObjCArray):
            if isinstance(objtype, ObjCArray):
                final = objtype
                found = True
            elif isinstance(objtype, ObjCStruct):
                found = True
            else:
                expr = ExprMem(expr, objtype.size * 8)
                found = True
        else:
            raise NotImplementedError('Unknown access' % node.expr)
        assert found
        out = (expr, objtype)
        return out

    def reduce_op_addr(self, node, **kwargs):
        if False:
            while True:
                i = 10
        'Reduce addr operator'
        if not node.expr.is_op('addr'):
            return None
        assert len(node.args) == 1
        out = []
        (src, src_type) = node.args[0].info
        void_type = self.types_mngr.void_ptr
        if isinstance(src_type, ObjCArray):
            out = (src.arg, ObjCPtr(src_type.objtype, void_type.align, void_type.size))
        elif isinstance(src, ExprMem):
            out = (src.ptr, ObjCPtr(src_type, void_type.align, void_type.size))
        elif isinstance(src_type, ObjCStruct):
            out = (src, ObjCPtr(src_type, void_type.align, void_type.size))
        elif isinstance(src_type, ObjCUnion):
            out = (src, ObjCPtr(src_type, void_type.align, void_type.size))
        else:
            raise NotImplementedError('unk type')
        return out

    def reduce_op_deref(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Reduce deref operator'
        if not node.expr.is_op('deref'):
            return None
        out = []
        (src, src_type) = node.args[0].info
        assert isinstance(src_type, (ObjCPtr, ObjCArray))
        void_type = self.types_mngr.void_ptr
        if isinstance(src_type, ObjCPtr):
            if isinstance(src_type.objtype, ObjCArray):
                size = void_type.size * 8
            else:
                size = src_type.objtype.size * 8
            out = (ExprMem(src, size), src_type.objtype)
        else:
            size = src_type.objtype.size * 8
            out = (ExprMem(src, size), src_type.objtype)
        return out
    reduction_rules = [reduce_known_expr, reduce_int, reduce_op_memberof, reduce_op_field, reduce_op_array, reduce_op_addr, reduce_op_deref]

    def get_expr(self, expr, c_context):
        if False:
            i = 10
            return i + 15
        'Translate a Miasm expression @expr (representing a C access) into a\n        tuple composed of a native Miasm expression and its C type.\n        @expr: Miasm expression (representing a C access)\n        @c_context: a dictionary linking known tokens (strings) to their\n        types. A token is linked to only one type.\n        '
        ret = self.reduce(expr, ctxt=c_context)
        if ret.info is None:
            return (None, None)
        return ret.info

class CTypesManager(object):
    """Represent a C object, without any layout information"""

    def __init__(self, types_ast, leaf_types):
        if False:
            print('Hello World!')
        self.types_ast = types_ast
        self.leaf_types = leaf_types

    @property
    def void_ptr(self):
        if False:
            while True:
                i = 10
        'Retrieve a void* objc'
        return self.leaf_types.types.get(CTypePtr(CTypeId('void')))

    @property
    def padding(self):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve a padding ctype'
        return CTypeId(PADDING_TYPE_NAME)

    def _get_objc(self, type_id, resolved=None, to_fix=None, lvl=0):
        if False:
            while True:
                i = 10
        if resolved is None:
            resolved = {}
        if to_fix is None:
            to_fix = []
        if type_id in resolved:
            return resolved[type_id]
        type_id = self.types_ast.get_type(type_id)
        fixed = True
        if isinstance(type_id, CTypeId):
            out = self.leaf_types.types.get(type_id, None)
            assert out is not None
        elif isinstance(type_id, CTypeUnion):
            args = []
            (align_max, size_max) = (0, 0)
            for (name, field) in type_id.fields:
                objc = self._get_objc(field, resolved, to_fix, lvl + 1)
                resolved[field] = objc
                align_max = max(align_max, objc.align)
                size_max = max(size_max, objc.size)
                args.append((name, objc, 0, objc.size))
            (align, size) = self.union_compute_align_size(align_max, size_max)
            out = ObjCUnion(type_id.name, align, size, args)
        elif isinstance(type_id, CTypeStruct):
            (align_max, size_max) = (0, 0)
            args = []
            (offset, align_max) = (0, 1)
            pad_index = 0
            for (name, field) in type_id.fields:
                objc = self._get_objc(field, resolved, to_fix, lvl + 1)
                resolved[field] = objc
                align_max = max(align_max, objc.align)
                new_offset = self.struct_compute_field_offset(objc, offset)
                if new_offset - offset:
                    pad_name = '__PAD__%d__' % pad_index
                    pad_index += 1
                    size = new_offset - offset
                    pad_objc = self._get_objc(CTypeArray(self.padding, size), resolved, to_fix, lvl + 1)
                    args.append((pad_name, pad_objc, offset, pad_objc.size))
                offset = new_offset
                args.append((name, objc, offset, objc.size))
                offset += objc.size
            (align, size) = self.struct_compute_align_size(align_max, offset)
            out = ObjCStruct(type_id.name, align, size, args)
        elif isinstance(type_id, CTypePtr):
            target = type_id.target
            out = ObjCPtr(None, self.void_ptr.align, self.void_ptr.size)
            fixed = False
        elif isinstance(type_id, CTypeArray):
            target = type_id.target
            objc = self._get_objc(target, resolved, to_fix, lvl + 1)
            resolved[target] = objc
            if type_id.size is None:
                out = ObjCPtr(objc, self.void_ptr.align, self.void_ptr.size)
            else:
                size = self.size_to_int(type_id.size)
                if size is None:
                    raise RuntimeError('Enable to compute objc size')
                else:
                    out = ObjCArray(objc, size)
            assert out.size is not None and out.align is not None
        elif isinstance(type_id, CTypeEnum):
            return self.leaf_types.types.get(CTypeId('int'))
        elif isinstance(type_id, CTypeFunc):
            type_ret = self._get_objc(type_id.type_ret, resolved, to_fix, lvl + 1)
            resolved[type_id.type_ret] = type_ret
            args = []
            for (name, arg) in type_id.args:
                objc = self._get_objc(arg, resolved, to_fix, lvl + 1)
                resolved[arg] = objc
                args.append((name, objc))
            out = ObjCFunc(type_id.name, type_id.abi, type_ret, args, self.void_ptr.align, self.void_ptr.size)
        elif isinstance(type_id, CTypeEllipsis):
            out = ObjCEllipsis()
        else:
            raise TypeError('Unknown type %r' % type_id.__class__)
        if not isinstance(out, ObjCEllipsis):
            assert out.align is not None and out.size is not None
        if fixed:
            resolved[type_id] = out
        else:
            to_fix.append((type_id, out))
        return out

    def get_objc(self, type_id):
        if False:
            print('Hello World!')
        'Get the ObjC corresponding to the CType @type_id\n        @type_id: CTypeBase instance'
        resolved = {}
        to_fix = []
        out = self._get_objc(type_id, resolved, to_fix)
        while to_fix:
            (type_id, objc_to_fix) = to_fix.pop()
            objc = self._get_objc(type_id.target, resolved, to_fix)
            objc_to_fix.objtype = objc
        self.check_objc(out)
        return out

    def check_objc(self, objc, done=None):
        if False:
            for i in range(10):
                print('nop')
        'Ensure each sub ObjC is resolved\n        @objc: ObjC instance'
        if done is None:
            done = set()
        if objc in done:
            return True
        done.add(objc)
        if isinstance(objc, (ObjCDecl, ObjCInt, ObjCEllipsis)):
            return True
        elif isinstance(objc, (ObjCPtr, ObjCArray)):
            assert self.check_objc(objc.objtype, done)
            return True
        elif isinstance(objc, (ObjCStruct, ObjCUnion)):
            for (_, field, _, _) in objc.fields:
                assert self.check_objc(field, done)
            return True
        elif isinstance(objc, ObjCFunc):
            assert self.check_objc(objc.type_ret, done)
            for (name, arg) in objc.args:
                assert self.check_objc(arg, done)
            return True
        else:
            assert False

    def size_to_int(self, size):
        if False:
            while True:
                i = 10
        'Resolve an array size\n        @size: CTypeOp or integer'
        if isinstance(size, CTypeOp):
            assert len(size.args) == 2
            (arg0, arg1) = [self.size_to_int(arg) for arg in size.args]
            if size.operator == '+':
                return arg0 + arg1
            elif size.operator == '-':
                return arg0 - arg1
            elif size.operator == '*':
                return arg0 * arg1
            elif size.operator == '/':
                return arg0 // arg1
            elif size.operator == '<<':
                return arg0 << arg1
            elif size.operator == '>>':
                return arg0 >> arg1
            else:
                raise ValueError('Unknown operator %s' % size.operator)
        elif isinstance(size, int_types):
            return size
        elif isinstance(size, CTypeSizeof):
            obj = self._get_objc(size.target)
            return obj.size
        else:
            raise TypeError('Unknown size type')

    def struct_compute_field_offset(self, obj, offset):
        if False:
            i = 10
            return i + 15
        'Compute the offset of the field @obj in the current structure'
        raise NotImplementedError('Abstract method')

    def struct_compute_align_size(self, align_max, size):
        if False:
            return 10
        'Compute the alignment and size of the current structure'
        raise NotImplementedError('Abstract method')

    def union_compute_align_size(self, align_max, size):
        if False:
            print('Hello World!')
        'Compute the alignment and size of the current union'
        raise NotImplementedError('Abstract method')

class CTypesManagerNotPacked(CTypesManager):
    """Store defined C types (not packed)"""

    def struct_compute_field_offset(self, obj, offset):
        if False:
            while True:
                i = 10
        'Compute the offset of the field @obj in the current structure\n        (not packed)'
        if obj.align > 1:
            offset = offset + obj.align - 1 & ~(obj.align - 1)
        return offset

    def struct_compute_align_size(self, align_max, size):
        if False:
            return 10
        'Compute the alignment and size of the current structure\n        (not packed)'
        if align_max > 1:
            size = size + align_max - 1 & ~(align_max - 1)
        return (align_max, size)

    def union_compute_align_size(self, align_max, size):
        if False:
            return 10
        'Compute the alignment and size of the current union\n        (not packed)'
        return (align_max, size)

class CTypesManagerPacked(CTypesManager):
    """Store defined C types (packed form)"""

    def struct_compute_field_offset(self, _, offset):
        if False:
            return 10
        'Compute the offset of the field @obj in the current structure\n        (packed form)'
        return offset

    def struct_compute_align_size(self, _, size):
        if False:
            i = 10
            return i + 15
        'Compute the alignment and size of the current structure\n        (packed form)'
        return (1, size)

    def union_compute_align_size(self, align_max, size):
        if False:
            for i in range(10):
                print('nop')
        'Compute the alignment and size of the current union\n        (packed form)'
        return (1, size)

class CHandler(object):
    """
    C manipulator for Miasm
    Miasm expr <-> C
    """
    exprCToExpr_cls = ExprCToExpr
    exprToAccessC_cls = ExprToAccessC

    def __init__(self, types_mngr, expr_types=None, C_types=None, simplify_c=access_simplifier, enforce_strict_access=True):
        if False:
            while True:
                i = 10
        self.exprc2expr = self.exprCToExpr_cls(expr_types, types_mngr)
        self.access_c_gen = self.exprToAccessC_cls(expr_types, types_mngr, enforce_strict_access)
        self.types_mngr = types_mngr
        self.simplify_c = simplify_c
        if expr_types is None:
            expr_types = {}
        self.expr_types = expr_types
        if C_types is None:
            C_types = {}
        self.C_types = C_types

    def updt_expr_types(self, expr_types):
        if False:
            while True:
                i = 10
        'Update expr_types\n        @expr_types: Dictionary associating name to type\n        '
        self.expr_types = expr_types
        self.exprc2expr.updt_expr_types(expr_types)
        self.access_c_gen.updt_expr_types(expr_types)

    def expr_to_c_access(self, expr, expr_context=None):
        if False:
            print('Hello World!')
        'Generate the C access object(s) for a given native Miasm expression.\n        @expr: Miasm expression\n        @expr_context: a dictionary linking known expressions to a set of types\n        '
        if expr_context is None:
            expr_context = self.expr_types
        return self.access_c_gen.get_accesses(expr, expr_context)

    def expr_to_c_and_types(self, expr, expr_context=None):
        if False:
            i = 10
            return i + 15
        'Generate the C access string and corresponding type for a given\n        native Miasm expression.\n        @expr_context: a dictionary linking known expressions to a set of types\n        '
        accesses = set()
        for access in self.expr_to_c_access(expr, expr_context):
            c_str = access_str(access.to_expr().visit(self.simplify_c))
            accesses.add((c_str, access.ctype))
        return accesses

    def expr_to_c(self, expr, expr_context=None):
        if False:
            while True:
                i = 10
        "Convert a Miasm @expr into it's C equivalent string\n        @expr_context: a dictionary linking known expressions to a set of types\n        "
        return set((access[0] for access in self.expr_to_c_and_types(expr, expr_context)))

    def expr_to_types(self, expr, expr_context=None):
        if False:
            for i in range(10):
                print('nop')
        'Get the possible types of the Miasm @expr\n        @expr_context: a dictionary linking known expressions to a set of types\n        '
        return set((access.ctype for access in self.expr_to_c_access(expr, expr_context)))

    def c_to_expr_and_type(self, c_str, c_context=None):
        if False:
            for i in range(10):
                print('nop')
        "Convert a C string expression to a Miasm expression and it's\n        corresponding c type\n        @c_str: C string\n        @c_context: (optional) dictionary linking known tokens (strings) to its\n        type.\n        "
        ast = parse_access(c_str)
        if c_context is None:
            c_context = self.C_types
        access_c = ast_get_c_access_expr(ast, c_context)
        return self.exprc2expr.get_expr(access_c, c_context)

    def c_to_expr(self, c_str, c_context=None):
        if False:
            return 10
        'Convert a C string expression to a Miasm expression\n        @c_str: C string\n        @c_context: (optional) dictionary linking known tokens (strings) to its\n        type.\n        '
        if c_context is None:
            c_context = self.C_types
        (expr, _) = self.c_to_expr_and_type(c_str, c_context)
        return expr

    def c_to_type(self, c_str, c_context=None):
        if False:
            print('Hello World!')
        'Get the type of a C string expression\n        @expr: Miasm expression\n        @c_context: (optional) dictionary linking known tokens (strings) to its\n        type.\n        '
        if c_context is None:
            c_context = self.C_types
        (_, ctype) = self.c_to_expr_and_type(c_str, c_context)
        return ctype

class CLeafTypes(object):
    """Define C types sizes/alignment for a given architecture"""
    pass