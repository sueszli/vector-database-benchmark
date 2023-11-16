import sys
from ..core._imperative_rt.core2 import pop_scope, push_scope, record_scope

class AutoNaming:
    """Name all executed operators automaticlly during tracing and record all tensors
    renamed by the user.
    """
    scopes = []
    c_ops = []
    name2ops = {}
    handle2names = {}
    __cls_attributes__ = {'scopes', 'c_ops', 'name2ops', 'handle2names'}

    @classmethod
    def clear(cls):
        if False:
            while True:
                i = 10
        for attr in cls.__cls_attributes__:
            getattr(cls, attr).clear()

    @classmethod
    def push_scope(cls, scope):
        if False:
            return 10
        if scope is not None:
            push_scope(scope)
            record_scope(sys._getframe().f_back.f_back, scope)
        cls.scopes.append(scope)

    @classmethod
    def pop_scope(cls):
        if False:
            for i in range(10):
                print('nop')
        scope = cls.scopes.pop()
        if scope is not None:
            pop_scope(scope)

    @classmethod
    def get_scope(cls):
        if False:
            i = 10
            return i + 15
        return '.'.join((s for s in cls.scopes if s is not None))

    @classmethod
    def gen_name(cls, x) -> str:
        if False:
            for i in range(10):
                print('nop')
        scope = cls.get_scope()
        name = x.c_name if x.c_name else x._name
        return scope + '.' + name if len(scope) else name

    @classmethod
    def record_var_name(cls, handle, name):
        if False:
            i = 10
            return i + 15
        cls.handle2names[handle] = name

    @classmethod
    def get_var_name(cls, handle):
        if False:
            while True:
                i = 10
        return cls.handle2names.pop(handle, None)

    @classmethod
    def record_opnode(cls, op):
        if False:
            while True:
                i = 10
        ops = cls.name2ops.get(op.name, [])
        if op not in ops:
            ops.append(op)
        cls.name2ops[op.name] = ops

    @classmethod
    def remove_duplicate_names(cls):
        if False:
            while True:
                i = 10
        for (key, ops) in cls.name2ops.items():
            if len(ops) == 1:
                continue
            for (i, op) in enumerate(ops):
                op.name = key + '[%s]' % str(i)
                if len(op.outputs) == 1:
                    continue
                for var in op.outputs:
                    var.name = var.name.replace(key, op.name)
        cls.name2ops.clear()