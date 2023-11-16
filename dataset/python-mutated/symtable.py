"""Interface to the compiler's internal symbol tables"""
import _symtable
from _symtable import USE, DEF_GLOBAL, DEF_NONLOCAL, DEF_LOCAL, DEF_PARAM, DEF_IMPORT, DEF_BOUND, DEF_ANNOT, SCOPE_OFF, SCOPE_MASK, FREE, LOCAL, GLOBAL_IMPLICIT, GLOBAL_EXPLICIT, CELL
import weakref
__all__ = ['symtable', 'SymbolTable', 'Class', 'Function', 'Symbol']

def symtable(code, filename, compile_type):
    if False:
        i = 10
        return i + 15
    ' Return the toplevel *SymbolTable* for the source code.\n\n    *filename* is the name of the file with the code\n    and *compile_type* is the *compile()* mode argument.\n    '
    top = _symtable.symtable(code, filename, compile_type)
    return _newSymbolTable(top, filename)

class SymbolTableFactory:

    def __init__(self):
        if False:
            return 10
        self.__memo = weakref.WeakValueDictionary()

    def new(self, table, filename):
        if False:
            for i in range(10):
                print('nop')
        if table.type == _symtable.TYPE_FUNCTION:
            return Function(table, filename)
        if table.type == _symtable.TYPE_CLASS:
            return Class(table, filename)
        return SymbolTable(table, filename)

    def __call__(self, table, filename):
        if False:
            print('Hello World!')
        key = (table, filename)
        obj = self.__memo.get(key, None)
        if obj is None:
            obj = self.__memo[key] = self.new(table, filename)
        return obj
_newSymbolTable = SymbolTableFactory()

class SymbolTable:

    def __init__(self, raw_table, filename):
        if False:
            i = 10
            return i + 15
        self._table = raw_table
        self._filename = filename
        self._symbols = {}

    def __repr__(self):
        if False:
            return 10
        if self.__class__ == SymbolTable:
            kind = ''
        else:
            kind = '%s ' % self.__class__.__name__
        if self._table.name == 'top':
            return '<{0}SymbolTable for module {1}>'.format(kind, self._filename)
        else:
            return '<{0}SymbolTable for {1} in {2}>'.format(kind, self._table.name, self._filename)

    def get_type(self):
        if False:
            print('Hello World!')
        "Return the type of the symbol table.\n\n        The values retuned are 'class', 'module' and\n        'function'.\n        "
        if self._table.type == _symtable.TYPE_MODULE:
            return 'module'
        if self._table.type == _symtable.TYPE_FUNCTION:
            return 'function'
        if self._table.type == _symtable.TYPE_CLASS:
            return 'class'
        assert self._table.type in (1, 2, 3), 'unexpected type: {0}'.format(self._table.type)

    def get_id(self):
        if False:
            return 10
        'Return an identifier for the table.\n        '
        return self._table.id

    def get_name(self):
        if False:
            for i in range(10):
                print('nop')
        "Return the table's name.\n\n        This corresponds to the name of the class, function\n        or 'top' if the table is for a class, function or\n        global respectively.\n        "
        return self._table.name

    def get_lineno(self):
        if False:
            i = 10
            return i + 15
        'Return the number of the first line in the\n        block for the table.\n        '
        return self._table.lineno

    def is_optimized(self):
        if False:
            print('Hello World!')
        'Return *True* if the locals in the table\n        are optimizable.\n        '
        return bool(self._table.type == _symtable.TYPE_FUNCTION)

    def is_nested(self):
        if False:
            print('Hello World!')
        'Return *True* if the block is a nested class\n        or function.'
        return bool(self._table.nested)

    def has_children(self):
        if False:
            return 10
        'Return *True* if the block has nested namespaces.\n        '
        return bool(self._table.children)

    def get_identifiers(self):
        if False:
            return 10
        'Return a list of names of symbols in the table.\n        '
        return self._table.symbols.keys()

    def lookup(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Lookup a *name* in the table.\n\n        Returns a *Symbol* instance.\n        '
        sym = self._symbols.get(name)
        if sym is None:
            flags = self._table.symbols[name]
            namespaces = self.__check_children(name)
            module_scope = self._table.name == 'top'
            sym = self._symbols[name] = Symbol(name, flags, namespaces, module_scope=module_scope)
        return sym

    def get_symbols(self):
        if False:
            return 10
        'Return a list of *Symbol* instances for\n        names in the table.\n        '
        return [self.lookup(ident) for ident in self.get_identifiers()]

    def __check_children(self, name):
        if False:
            return 10
        return [_newSymbolTable(st, self._filename) for st in self._table.children if st.name == name]

    def get_children(self):
        if False:
            i = 10
            return i + 15
        'Return a list of the nested symbol tables.\n        '
        return [_newSymbolTable(st, self._filename) for st in self._table.children]

class Function(SymbolTable):
    __params = None
    __locals = None
    __frees = None
    __globals = None
    __nonlocals = None

    def __idents_matching(self, test_func):
        if False:
            for i in range(10):
                print('nop')
        return tuple((ident for ident in self.get_identifiers() if test_func(self._table.symbols[ident])))

    def get_parameters(self):
        if False:
            while True:
                i = 10
        'Return a tuple of parameters to the function.\n        '
        if self.__params is None:
            self.__params = self.__idents_matching(lambda x: x & DEF_PARAM)
        return self.__params

    def get_locals(self):
        if False:
            while True:
                i = 10
        'Return a tuple of locals in the function.\n        '
        if self.__locals is None:
            locs = (LOCAL, CELL)
            test = lambda x: x >> SCOPE_OFF & SCOPE_MASK in locs
            self.__locals = self.__idents_matching(test)
        return self.__locals

    def get_globals(self):
        if False:
            print('Hello World!')
        'Return a tuple of globals in the function.\n        '
        if self.__globals is None:
            glob = (GLOBAL_IMPLICIT, GLOBAL_EXPLICIT)
            test = lambda x: x >> SCOPE_OFF & SCOPE_MASK in glob
            self.__globals = self.__idents_matching(test)
        return self.__globals

    def get_nonlocals(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a tuple of nonlocals in the function.\n        '
        if self.__nonlocals is None:
            self.__nonlocals = self.__idents_matching(lambda x: x & DEF_NONLOCAL)
        return self.__nonlocals

    def get_frees(self):
        if False:
            while True:
                i = 10
        'Return a tuple of free variables in the function.\n        '
        if self.__frees is None:
            is_free = lambda x: x >> SCOPE_OFF & SCOPE_MASK == FREE
            self.__frees = self.__idents_matching(is_free)
        return self.__frees

class Class(SymbolTable):
    __methods = None

    def get_methods(self):
        if False:
            return 10
        'Return a tuple of methods declared in the class.\n        '
        if self.__methods is None:
            d = {}
            for st in self._table.children:
                d[st.name] = 1
            self.__methods = tuple(d)
        return self.__methods

class Symbol:

    def __init__(self, name, flags, namespaces=None, *, module_scope=False):
        if False:
            for i in range(10):
                print('nop')
        self.__name = name
        self.__flags = flags
        self.__scope = flags >> SCOPE_OFF & SCOPE_MASK
        self.__namespaces = namespaces or ()
        self.__module_scope = module_scope

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<symbol {0!r}>'.format(self.__name)

    def get_name(self):
        if False:
            while True:
                i = 10
        'Return a name of a symbol.\n        '
        return self.__name

    def is_referenced(self):
        if False:
            i = 10
            return i + 15
        'Return *True* if the symbol is used in\n        its block.\n        '
        return bool(self.__flags & _symtable.USE)

    def is_parameter(self):
        if False:
            return 10
        'Return *True* if the symbol is a parameter.\n        '
        return bool(self.__flags & DEF_PARAM)

    def is_global(self):
        if False:
            print('Hello World!')
        'Return *True* if the sysmbol is global.\n        '
        return bool(self.__scope in (GLOBAL_IMPLICIT, GLOBAL_EXPLICIT) or (self.__module_scope and self.__flags & DEF_BOUND))

    def is_nonlocal(self):
        if False:
            for i in range(10):
                print('nop')
        'Return *True* if the symbol is nonlocal.'
        return bool(self.__flags & DEF_NONLOCAL)

    def is_declared_global(self):
        if False:
            i = 10
            return i + 15
        'Return *True* if the symbol is declared global\n        with a global statement.'
        return bool(self.__scope == GLOBAL_EXPLICIT)

    def is_local(self):
        if False:
            i = 10
            return i + 15
        'Return *True* if the symbol is local.\n        '
        return bool(self.__scope in (LOCAL, CELL) or (self.__module_scope and self.__flags & DEF_BOUND))

    def is_annotated(self):
        if False:
            i = 10
            return i + 15
        'Return *True* if the symbol is annotated.\n        '
        return bool(self.__flags & DEF_ANNOT)

    def is_free(self):
        if False:
            while True:
                i = 10
        'Return *True* if a referenced symbol is\n        not assigned to.\n        '
        return bool(self.__scope == FREE)

    def is_imported(self):
        if False:
            return 10
        'Return *True* if the symbol is created from\n        an import statement.\n        '
        return bool(self.__flags & DEF_IMPORT)

    def is_assigned(self):
        if False:
            for i in range(10):
                print('nop')
        'Return *True* if a symbol is assigned to.'
        return bool(self.__flags & DEF_LOCAL)

    def is_namespace(self):
        if False:
            while True:
                i = 10
        'Returns *True* if name binding introduces new namespace.\n\n        If the name is used as the target of a function or class\n        statement, this will be true.\n\n        Note that a single name can be bound to multiple objects.  If\n        is_namespace() is true, the name may also be bound to other\n        objects, like an int or list, that does not introduce a new\n        namespace.\n        '
        return bool(self.__namespaces)

    def get_namespaces(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of namespaces bound to this name'
        return self.__namespaces

    def get_namespace(self):
        if False:
            print('Hello World!')
        'Return the single namespace bound to this name.\n\n        Raises ValueError if the name is bound to multiple namespaces.\n        '
        if len(self.__namespaces) != 1:
            raise ValueError('name is bound to multiple namespaces')
        return self.__namespaces[0]
if __name__ == '__main__':
    import os, sys
    with open(sys.argv[0]) as f:
        src = f.read()
    mod = symtable(src, os.path.split(sys.argv[0])[1], 'exec')
    for ident in mod.get_identifiers():
        info = mod.lookup(ident)
        print(info, info.is_local(), info.is_namespace())