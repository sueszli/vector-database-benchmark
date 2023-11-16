from collections import OrderedDict
from textwrap import dedent
import operator
from . import ExprNodes
from . import Nodes
from . import PyrexTypes
from . import Builtin
from . import Naming
from .Errors import error, warning
from .Code import UtilityCode, TempitaUtilityCode, PyxCodeWriter
from .Visitor import VisitorTransform
from .StringEncoding import EncodedString
from .TreeFragment import TreeFragment
from .ParseTreeTransforms import NormalizeTree, SkipDeclarations
from .Options import copy_inherited_directives
_dataclass_loader_utilitycode = None

def make_dataclasses_module_callnode(pos):
    if False:
        return 10
    global _dataclass_loader_utilitycode
    if not _dataclass_loader_utilitycode:
        python_utility_code = UtilityCode.load_cached('Dataclasses_fallback', 'Dataclasses.py')
        python_utility_code = EncodedString(python_utility_code.impl)
        _dataclass_loader_utilitycode = TempitaUtilityCode.load('SpecificModuleLoader', 'Dataclasses.c', context={'cname': 'dataclasses', 'py_code': python_utility_code.as_c_string_literal()})
    return ExprNodes.PythonCapiCallNode(pos, '__Pyx_Load_dataclasses_Module', PyrexTypes.CFuncType(PyrexTypes.py_object_type, []), utility_code=_dataclass_loader_utilitycode, args=[])

def make_dataclass_call_helper(pos, callable, kwds):
    if False:
        return 10
    utility_code = UtilityCode.load_cached('DataclassesCallHelper', 'Dataclasses.c')
    func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('callable', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('kwds', PyrexTypes.py_object_type, None)])
    return ExprNodes.PythonCapiCallNode(pos, function_name='__Pyx_DataclassesCallHelper', func_type=func_type, utility_code=utility_code, args=[callable, kwds])

class RemoveAssignmentsToNames(VisitorTransform, SkipDeclarations):
    """
    Cython (and Python) normally treats

    class A:
         x = 1

    as generating a class attribute. However for dataclasses the `= 1` should be interpreted as
    a default value to initialize an instance attribute with.
    This transform therefore removes the `x=1` assignment so that the class attribute isn't
    generated, while recording what it has removed so that it can be used in the initialization.
    """

    def __init__(self, names):
        if False:
            for i in range(10):
                print('nop')
        super(RemoveAssignmentsToNames, self).__init__()
        self.names = names
        self.removed_assignments = {}

    def visit_CClassNode(self, node):
        if False:
            return 10
        self.visitchildren(node)
        return node

    def visit_PyClassNode(self, node):
        if False:
            print('Hello World!')
        return node

    def visit_FuncDefNode(self, node):
        if False:
            i = 10
            return i + 15
        return node

    def visit_SingleAssignmentNode(self, node):
        if False:
            return 10
        if node.lhs.is_name and node.lhs.name in self.names:
            if node.lhs.name in self.removed_assignments:
                warning(node.pos, "Multiple assignments for '%s' in dataclass; using most recent" % node.lhs.name, 1)
            self.removed_assignments[node.lhs.name] = node.rhs
            return []
        return node

    def visit_Node(self, node):
        if False:
            while True:
                i = 10
        self.visitchildren(node)
        return node

class TemplateCode(object):
    """
    Adds the ability to keep track of placeholder argument names to PyxCodeWriter.

    Also adds extra_stats which are nodes bundled at the end when this
    is converted to a tree.
    """
    _placeholder_count = 0

    def __init__(self, writer=None, placeholders=None, extra_stats=None):
        if False:
            return 10
        self.writer = PyxCodeWriter() if writer is None else writer
        self.placeholders = {} if placeholders is None else placeholders
        self.extra_stats = [] if extra_stats is None else extra_stats

    def add_code_line(self, code_line):
        if False:
            print('Hello World!')
        self.writer.putln(code_line)

    def add_code_lines(self, code_lines):
        if False:
            for i in range(10):
                print('nop')
        for line in code_lines:
            self.writer.putln(line)

    def reset(self):
        if False:
            return 10
        self.writer.reset()

    def empty(self):
        if False:
            i = 10
            return i + 15
        return self.writer.empty()

    def indenter(self):
        if False:
            return 10
        return self.writer.indenter()

    def new_placeholder(self, field_names, value):
        if False:
            for i in range(10):
                print('nop')
        name = self._new_placeholder_name(field_names)
        self.placeholders[name] = value
        return name

    def add_extra_statements(self, statements):
        if False:
            for i in range(10):
                print('nop')
        if self.extra_stats is None:
            assert False, 'Can only use add_extra_statements on top-level writer'
        self.extra_stats.extend(statements)

    def _new_placeholder_name(self, field_names):
        if False:
            i = 10
            return i + 15
        while True:
            name = 'DATACLASS_PLACEHOLDER_%d' % self._placeholder_count
            if name not in self.placeholders and name not in field_names:
                break
            self._placeholder_count += 1
        return name

    def generate_tree(self, level='c_class'):
        if False:
            print('Hello World!')
        stat_list_node = TreeFragment(self.writer.getvalue(), level=level, pipeline=[NormalizeTree(None)]).substitute(self.placeholders)
        stat_list_node.stats += self.extra_stats
        return stat_list_node

    def insertion_point(self):
        if False:
            for i in range(10):
                print('nop')
        new_writer = self.writer.insertion_point()
        return TemplateCode(writer=new_writer, placeholders=self.placeholders, extra_stats=self.extra_stats)

class _MISSING_TYPE(object):
    pass
MISSING = _MISSING_TYPE()

class Field(object):
    """
    Field is based on the dataclasses.field class from the standard library module.
    It is used internally during the generation of Cython dataclasses to keep track
    of the settings for individual attributes.

    Attributes of this class are stored as nodes so they can be used in code construction
    more readily (i.e. we store BoolNode rather than bool)
    """
    default = MISSING
    default_factory = MISSING
    private = False
    literal_keys = ('repr', 'hash', 'init', 'compare', 'metadata')

    def __init__(self, pos, default=MISSING, default_factory=MISSING, repr=None, hash=None, init=None, compare=None, metadata=None, is_initvar=False, is_classvar=False, **additional_kwds):
        if False:
            print('Hello World!')
        if default is not MISSING:
            self.default = default
        if default_factory is not MISSING:
            self.default_factory = default_factory
        self.repr = repr or ExprNodes.BoolNode(pos, value=True)
        self.hash = hash or ExprNodes.NoneNode(pos)
        self.init = init or ExprNodes.BoolNode(pos, value=True)
        self.compare = compare or ExprNodes.BoolNode(pos, value=True)
        self.metadata = metadata or ExprNodes.NoneNode(pos)
        self.is_initvar = is_initvar
        self.is_classvar = is_classvar
        for (k, v) in additional_kwds.items():
            error(v.pos, "cython.dataclasses.field() got an unexpected keyword argument '%s'" % k)
        for field_name in self.literal_keys:
            field_value = getattr(self, field_name)
            if not field_value.is_literal:
                error(field_value.pos, "cython.dataclasses.field parameter '%s' must be a literal value" % field_name)

    def iterate_record_node_arguments(self):
        if False:
            print('Hello World!')
        for key in self.literal_keys + ('default', 'default_factory'):
            value = getattr(self, key)
            if value is not MISSING:
                yield (key, value)

def process_class_get_fields(node):
    if False:
        while True:
            i = 10
    var_entries = node.scope.var_entries
    var_entries = sorted(var_entries, key=operator.attrgetter('pos'))
    var_names = [entry.name for entry in var_entries]
    transform = RemoveAssignmentsToNames(var_names)
    transform(node)
    default_value_assignments = transform.removed_assignments
    base_type = node.base_type
    fields = OrderedDict()
    while base_type:
        if base_type.is_external or not base_type.scope.implemented:
            warning(node.pos, 'Cannot reliably handle Cython dataclasses with base types in external modules since it is not possible to tell what fields they have', 2)
        if base_type.dataclass_fields:
            fields = base_type.dataclass_fields.copy()
            break
        base_type = base_type.base_type
    for entry in var_entries:
        name = entry.name
        is_initvar = entry.declared_with_pytyping_modifier('dataclasses.InitVar')
        is_classvar = entry.declared_with_pytyping_modifier('typing.ClassVar')
        if name in default_value_assignments:
            assignment = default_value_assignments[name]
            if isinstance(assignment, ExprNodes.CallNode) and (assignment.function.as_cython_attribute() == 'dataclasses.field' or Builtin.exprnode_to_known_standard_library_name(assignment.function, node.scope) == 'dataclasses.field'):
                valid_general_call = isinstance(assignment, ExprNodes.GeneralCallNode) and isinstance(assignment.positional_args, ExprNodes.TupleNode) and (not assignment.positional_args.args) and (assignment.keyword_args is None or isinstance(assignment.keyword_args, ExprNodes.DictNode))
                valid_simple_call = isinstance(assignment, ExprNodes.SimpleCallNode) and (not assignment.args)
                if not (valid_general_call or valid_simple_call):
                    error(assignment.pos, "Call to 'cython.dataclasses.field' must only consist of compile-time keyword arguments")
                    continue
                keyword_args = assignment.keyword_args.as_python_dict() if valid_general_call and assignment.keyword_args else {}
                if 'default' in keyword_args and 'default_factory' in keyword_args:
                    error(assignment.pos, 'cannot specify both default and default_factory')
                    continue
                field = Field(node.pos, **keyword_args)
            else:
                if assignment.type in [Builtin.list_type, Builtin.dict_type, Builtin.set_type]:
                    error(assignment.pos, "mutable default <class '{0}'> for field {1} is not allowed: use default_factory".format(assignment.type.name, name))
                field = Field(node.pos, default=assignment)
        else:
            field = Field(node.pos)
        field.is_initvar = is_initvar
        field.is_classvar = is_classvar
        if entry.visibility == 'private':
            field.private = True
        fields[name] = field
    node.entry.type.dataclass_fields = fields
    return fields

def handle_cclass_dataclass(node, dataclass_args, analyse_decs_transform):
    if False:
        while True:
            i = 10
    kwargs = dict(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, kw_only=False)
    if dataclass_args is not None:
        if dataclass_args[0]:
            error(node.pos, 'cython.dataclasses.dataclass takes no positional arguments')
        for (k, v) in dataclass_args[1].items():
            if k not in kwargs:
                error(node.pos, "cython.dataclasses.dataclass() got an unexpected keyword argument '%s'" % k)
            if not isinstance(v, ExprNodes.BoolNode):
                error(node.pos, 'Arguments passed to cython.dataclasses.dataclass must be True or False')
            kwargs[k] = v.value
    kw_only = kwargs['kw_only']
    fields = process_class_get_fields(node)
    dataclass_module = make_dataclasses_module_callnode(node.pos)
    dataclass_params_func = ExprNodes.AttributeNode(node.pos, obj=dataclass_module, attribute=EncodedString('_DataclassParams'))
    dataclass_params_keywords = ExprNodes.DictNode.from_pairs(node.pos, [(ExprNodes.IdentifierStringNode(node.pos, value=EncodedString(k)), ExprNodes.BoolNode(node.pos, value=v)) for (k, v) in kwargs.items()] + [(ExprNodes.IdentifierStringNode(node.pos, value=EncodedString(k)), ExprNodes.BoolNode(node.pos, value=v)) for (k, v) in [('kw_only', kw_only), ('match_args', False), ('slots', False), ('weakref_slot', False)]])
    dataclass_params = make_dataclass_call_helper(node.pos, dataclass_params_func, dataclass_params_keywords)
    dataclass_params_assignment = Nodes.SingleAssignmentNode(node.pos, lhs=ExprNodes.NameNode(node.pos, name=EncodedString('__dataclass_params__')), rhs=dataclass_params)
    dataclass_fields_stats = _set_up_dataclass_fields(node, fields, dataclass_module)
    stats = Nodes.StatListNode(node.pos, stats=[dataclass_params_assignment] + dataclass_fields_stats)
    code = TemplateCode()
    generate_init_code(code, kwargs['init'], node, fields, kw_only)
    generate_repr_code(code, kwargs['repr'], node, fields)
    generate_eq_code(code, kwargs['eq'], node, fields)
    generate_order_code(code, kwargs['order'], node, fields)
    generate_hash_code(code, kwargs['unsafe_hash'], kwargs['eq'], kwargs['frozen'], node, fields)
    stats.stats += code.generate_tree().stats
    comp_directives = Nodes.CompilerDirectivesNode(node.pos, directives=copy_inherited_directives(node.scope.directives, annotation_typing=False), body=stats)
    comp_directives.analyse_declarations(node.scope)
    analyse_decs_transform.enter_scope(node, node.scope)
    analyse_decs_transform.visit(comp_directives)
    analyse_decs_transform.exit_scope()
    node.body.stats.append(comp_directives)

def generate_init_code(code, init, node, fields, kw_only):
    if False:
        for i in range(10):
            print('nop')
    '\n    Notes on CPython generated "__init__":\n    * Implemented in `_init_fn`.\n    * The use of the `dataclasses._HAS_DEFAULT_FACTORY` sentinel value as\n      the default argument for fields that need constructing with a factory\n      function is copied from the CPython implementation. (`None` isn\'t\n      suitable because it could also be a value for the user to pass.)\n      There\'s no real reason why it needs importing from the dataclasses module\n      though - it could equally be a value generated by Cython when the module loads.\n    * seen_default and the associated error message are copied directly from Python\n    * Call to user-defined __post_init__ function (if it exists) is copied from\n      CPython.\n\n    Cython behaviour deviates a little here (to be decided if this is right...)\n    Because the class variable from the assignment does not exist Cython fields will\n    return None (or whatever their type default is) if not initialized while Python\n    dataclasses will fall back to looking up the class variable.\n    '
    if not init or node.scope.lookup_here('__init__'):
        return
    selfname = '__dataclass_self__' if 'self' in fields else 'self'
    args = [selfname]
    if kw_only:
        args.append('*')
    function_start_point = code.insertion_point()
    code = code.insertion_point()
    dataclass_module = make_dataclasses_module_callnode(node.pos)
    has_default_factory = ExprNodes.AttributeNode(node.pos, obj=dataclass_module, attribute=EncodedString('_HAS_DEFAULT_FACTORY'))
    default_factory_placeholder = code.new_placeholder(fields, has_default_factory)
    seen_default = False
    for (name, field) in fields.items():
        entry = node.scope.lookup(name)
        if entry.annotation:
            annotation = u': %s' % entry.annotation.string.value
        else:
            annotation = u''
        assignment = u''
        if field.default is not MISSING or field.default_factory is not MISSING:
            seen_default = True
            if field.default_factory is not MISSING:
                ph_name = default_factory_placeholder
            else:
                ph_name = code.new_placeholder(fields, field.default)
            assignment = u' = %s' % ph_name
        elif seen_default and (not kw_only) and field.init.value:
            error(entry.pos, "non-default argument '%s' follows default argument in dataclass __init__" % name)
            code.reset()
            return
        if field.init.value:
            args.append(u'%s%s%s' % (name, annotation, assignment))
        if field.is_initvar:
            continue
        elif field.default_factory is MISSING:
            if field.init.value:
                code.add_code_line(u'    %s.%s = %s' % (selfname, name, name))
            elif assignment:
                code.add_code_line(u'    %s.%s%s' % (selfname, name, assignment))
        else:
            ph_name = code.new_placeholder(fields, field.default_factory)
            if field.init.value:
                code.add_code_line(u'    %s.%s = %s() if %s is %s else %s' % (selfname, name, ph_name, name, default_factory_placeholder, name))
            else:
                code.add_code_line(u'    %s.%s = %s()' % (selfname, name, ph_name))
    if node.scope.lookup('__post_init__'):
        post_init_vars = ', '.join((name for (name, field) in fields.items() if field.is_initvar))
        code.add_code_line('    %s.__post_init__(%s)' % (selfname, post_init_vars))
    if code.empty():
        code.add_code_line('    pass')
    args = u', '.join(args)
    function_start_point.add_code_line(u'def __init__(%s):' % args)

def generate_repr_code(code, repr, node, fields):
    if False:
        while True:
            i = 10
    '\n    The core of the CPython implementation is just:\n    [\'return self.__class__.__qualname__ + f"(\' +\n                     \', \'.join([f"{f.name}={{self.{f.name}!r}}"\n                                for f in fields]) +\n                     \')"\'],\n\n    The only notable difference here is self.__class__.__qualname__ -> type(self).__name__\n    which is because Cython currently supports Python 2.\n\n    However, it also has some guards for recursive repr invocations. In the standard\n    library implementation they\'re done with a wrapper decorator that captures a set\n    (with the set keyed by id and thread). Here we create a set as a thread local\n    variable and key only by id.\n    '
    if not repr or node.scope.lookup('__repr__'):
        return
    needs_recursive_guard = False
    for name in fields.keys():
        entry = node.scope.lookup(name)
        type_ = entry.type
        if type_.is_memoryviewslice:
            type_ = type_.dtype
        if not type_.is_pyobject:
            continue
        if not type_.is_gc_simple:
            needs_recursive_guard = True
            break
    if needs_recursive_guard:
        code.add_code_line("__pyx_recursive_repr_guard = __import__('threading').local()")
        code.add_code_line('__pyx_recursive_repr_guard.running = set()')
    code.add_code_line('def __repr__(self):')
    if needs_recursive_guard:
        code.add_code_line('    key = id(self)')
        code.add_code_line('    guard_set = self.__pyx_recursive_repr_guard.running')
        code.add_code_line("    if key in guard_set: return '...'")
        code.add_code_line('    guard_set.add(key)')
        code.add_code_line('    try:')
    strs = [u'%s={self.%s!r}' % (name, name) for (name, field) in fields.items() if field.repr.value and (not field.is_initvar)]
    format_string = u', '.join(strs)
    code.add_code_line(u'        name = getattr(type(self), "__qualname__", type(self).__name__)')
    code.add_code_line(u"        return f'{name}(%s)'" % format_string)
    if needs_recursive_guard:
        code.add_code_line('    finally:')
        code.add_code_line('        guard_set.remove(key)')

def generate_cmp_code(code, op, funcname, node, fields):
    if False:
        for i in range(10):
            print('nop')
    if node.scope.lookup_here(funcname):
        return
    names = [name for (name, field) in fields.items() if field.compare.value and (not field.is_initvar)]
    code.add_code_lines(['def %s(self, other):' % funcname, '    if not isinstance(other, %s):' % node.class_name, '        return NotImplemented', '    cdef %s other_cast' % node.class_name, '    other_cast = <%s>other' % node.class_name])
    checks = []
    for name in names:
        checks.append('(self.%s %s other_cast.%s)' % (name, op, name))
    if checks:
        code.add_code_line('    return ' + ' and '.join(checks))
    elif '=' in op:
        code.add_code_line('    return True')
    else:
        code.add_code_line('    return False')

def generate_eq_code(code, eq, node, fields):
    if False:
        i = 10
        return i + 15
    if not eq:
        return
    generate_cmp_code(code, '==', '__eq__', node, fields)

def generate_order_code(code, order, node, fields):
    if False:
        i = 10
        return i + 15
    if not order:
        return
    for (op, name) in [('<', '__lt__'), ('<=', '__le__'), ('>', '__gt__'), ('>=', '__ge__')]:
        generate_cmp_code(code, op, name, node, fields)

def generate_hash_code(code, unsafe_hash, eq, frozen, node, fields):
    if False:
        for i in range(10):
            print('nop')
    "\n    Copied from CPython implementation - the intention is to follow this as far as\n    is possible:\n    #    +------------------- unsafe_hash= parameter\n    #    |       +----------- eq= parameter\n    #    |       |       +--- frozen= parameter\n    #    |       |       |\n    #    v       v       v    |        |        |\n    #                         |   no   |  yes   |  <--- class has explicitly defined __hash__\n    # +=======+=======+=======+========+========+\n    # | False | False | False |        |        | No __eq__, use the base class __hash__\n    # +-------+-------+-------+--------+--------+\n    # | False | False | True  |        |        | No __eq__, use the base class __hash__\n    # +-------+-------+-------+--------+--------+\n    # | False | True  | False | None   |        | <-- the default, not hashable\n    # +-------+-------+-------+--------+--------+\n    # | False | True  | True  | add    |        | Frozen, so hashable, allows override\n    # +-------+-------+-------+--------+--------+\n    # | True  | False | False | add    | raise  | Has no __eq__, but hashable\n    # +-------+-------+-------+--------+--------+\n    # | True  | False | True  | add    | raise  | Has no __eq__, but hashable\n    # +-------+-------+-------+--------+--------+\n    # | True  | True  | False | add    | raise  | Not frozen, but hashable\n    # +-------+-------+-------+--------+--------+\n    # | True  | True  | True  | add    | raise  | Frozen, so hashable\n    # +=======+=======+=======+========+========+\n    # For boxes that are blank, __hash__ is untouched and therefore\n    # inherited from the base class.  If the base is object, then\n    # id-based hashing is used.\n\n    The Python implementation creates a tuple of all the fields, then hashes them.\n    This implementation creates a tuple of all the hashes of all the fields and hashes that.\n    The reason for this slight difference is to avoid to-Python conversions for anything\n    that Cython knows how to hash directly (It doesn't look like this currently applies to\n    anything though...).\n    "
    hash_entry = node.scope.lookup_here('__hash__')
    if hash_entry:
        if unsafe_hash:
            error(node.pos, 'Cannot overwrite attribute __hash__ in class %s' % node.class_name)
        return
    if not unsafe_hash:
        if not eq:
            return
        if not frozen:
            code.add_extra_statements([Nodes.SingleAssignmentNode(node.pos, lhs=ExprNodes.NameNode(node.pos, name=EncodedString('__hash__')), rhs=ExprNodes.NoneNode(node.pos))])
            return
    names = [name for (name, field) in fields.items() if not field.is_initvar and (field.compare.value if field.hash.value is None else field.hash.value)]
    hash_tuple_items = u', '.join((u'self.%s' % name for name in names))
    if hash_tuple_items:
        hash_tuple_items += u','
    code.add_code_lines(['def __hash__(self):', '    return hash((%s))' % hash_tuple_items])

def get_field_type(pos, entry):
    if False:
        while True:
            i = 10
    '\n    sets the .type attribute for a field\n\n    Returns the annotation if possible (since this is what the dataclasses\n    module does). If not (for example, attributes defined with cdef) then\n    it creates a string fallback.\n    '
    if entry.annotation:
        return entry.annotation.string
    else:
        s = EncodedString(entry.type.declaration_code('', for_display=1))
        return ExprNodes.StringNode(pos, value=s)

class FieldRecordNode(ExprNodes.ExprNode):
    """
    __dataclass_fields__ contains a bunch of field objects recording how each field
    of the dataclass was initialized (mainly corresponding to the arguments passed to
    the "field" function). This node is used for the attributes of these field objects.

    If possible, coerces `arg` to a Python object.
    Otherwise, generates a sensible backup string.
    """
    subexprs = ['arg']

    def __init__(self, pos, arg):
        if False:
            return 10
        super(FieldRecordNode, self).__init__(pos, arg=arg)

    def analyse_types(self, env):
        if False:
            print('Hello World!')
        self.arg.analyse_types(env)
        self.type = self.arg.type
        return self

    def coerce_to_pyobject(self, env):
        if False:
            print('Hello World!')
        if self.arg.type.can_coerce_to_pyobject(env):
            return self.arg.coerce_to_pyobject(env)
        else:
            return self._make_string()

    def _make_string(self):
        if False:
            for i in range(10):
                print('nop')
        from .AutoDocTransforms import AnnotationWriter
        writer = AnnotationWriter(description='Dataclass field')
        string = writer.write(self.arg)
        return ExprNodes.StringNode(self.pos, value=EncodedString(string))

    def generate_evaluation_code(self, code):
        if False:
            print('Hello World!')
        return self.arg.generate_evaluation_code(code)

def _set_up_dataclass_fields(node, fields, dataclass_module):
    if False:
        return 10
    variables_assignment_stats = []
    for (name, field) in fields.items():
        if field.private:
            continue
        for attrname in ['default', 'default_factory']:
            field_default = getattr(field, attrname)
            if field_default is MISSING or field_default.is_literal or field_default.is_name:
                continue
            global_scope = node.scope.global_scope()
            module_field_name = global_scope.mangle(global_scope.mangle(Naming.dataclass_field_default_cname, node.class_name), name)
            field_node = ExprNodes.NameNode(field_default.pos, name=EncodedString(module_field_name))
            field_node.entry = global_scope.declare_var(field_node.name, type=field_default.type or PyrexTypes.unspecified_type, pos=field_default.pos, cname=field_node.name, is_cdef=True)
            setattr(field, attrname, field_node)
            variables_assignment_stats.append(Nodes.SingleAssignmentNode(field_default.pos, lhs=field_node, rhs=field_default))
    placeholders = {}
    field_func = ExprNodes.AttributeNode(node.pos, obj=dataclass_module, attribute=EncodedString('field'))
    dc_fields = ExprNodes.DictNode(node.pos, key_value_pairs=[])
    dc_fields_namevalue_assignments = []
    for (name, field) in fields.items():
        if field.private:
            continue
        type_placeholder_name = 'PLACEHOLDER_%s' % name
        placeholders[type_placeholder_name] = get_field_type(node.pos, node.scope.entries[name])
        field_type_placeholder_name = 'PLACEHOLDER_FIELD_TYPE_%s' % name
        if field.is_initvar:
            placeholders[field_type_placeholder_name] = ExprNodes.AttributeNode(node.pos, obj=dataclass_module, attribute=EncodedString('_FIELD_INITVAR'))
        elif field.is_classvar:
            placeholders[field_type_placeholder_name] = ExprNodes.AttributeNode(node.pos, obj=dataclass_module, attribute=EncodedString('_FIELD_CLASSVAR'))
        else:
            placeholders[field_type_placeholder_name] = ExprNodes.AttributeNode(node.pos, obj=dataclass_module, attribute=EncodedString('_FIELD'))
        dc_field_keywords = ExprNodes.DictNode.from_pairs(node.pos, [(ExprNodes.IdentifierStringNode(node.pos, value=EncodedString(k)), FieldRecordNode(node.pos, arg=v)) for (k, v) in field.iterate_record_node_arguments()])
        dc_field_call = make_dataclass_call_helper(node.pos, field_func, dc_field_keywords)
        dc_fields.key_value_pairs.append(ExprNodes.DictItemNode(node.pos, key=ExprNodes.IdentifierStringNode(node.pos, value=EncodedString(name)), value=dc_field_call))
        dc_fields_namevalue_assignments.append(dedent(u'                __dataclass_fields__[{0!r}].name = {0!r}\n                __dataclass_fields__[{0!r}].type = {1}\n                __dataclass_fields__[{0!r}]._field_type = {2}\n            ').format(name, type_placeholder_name, field_type_placeholder_name))
    dataclass_fields_assignment = Nodes.SingleAssignmentNode(node.pos, lhs=ExprNodes.NameNode(node.pos, name=EncodedString('__dataclass_fields__')), rhs=dc_fields)
    dc_fields_namevalue_assignments = u'\n'.join(dc_fields_namevalue_assignments)
    dc_fields_namevalue_assignments = TreeFragment(dc_fields_namevalue_assignments, level='c_class', pipeline=[NormalizeTree(None)])
    dc_fields_namevalue_assignments = dc_fields_namevalue_assignments.substitute(placeholders)
    return variables_assignment_stats + [dataclass_fields_assignment] + dc_fields_namevalue_assignments.stats