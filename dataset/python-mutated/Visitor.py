from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
cython.declare(_PRINTABLE=tuple)
if sys.version_info[0] >= 3:
    _PRINTABLE = (bytes, str, int, float)
else:
    _PRINTABLE = (str, unicode, long, int, float)

class TreeVisitor(object):
    """
    Base class for writing visitors for a Cython tree, contains utilities for
    recursing such trees using visitors. Each node is
    expected to have a child_attrs iterable containing the names of attributes
    containing child nodes or lists of child nodes. Lists are not considered
    part of the tree structure (i.e. contained nodes are considered direct
    children of the parent node).

    visit_children visits each of the children of a given node (see the visit_children
    documentation). When recursing the tree using visit_children, an attribute
    access_path is maintained which gives information about the current location
    in the tree as a stack of tuples: (parent_node, attrname, index), representing
    the node, attribute and optional list index that was taken in each step in the path to
    the current node.

    Example:

    >>> class SampleNode(object):
    ...     child_attrs = ["head", "body"]
    ...     def __init__(self, value, head=None, body=None):
    ...         self.value = value
    ...         self.head = head
    ...         self.body = body
    ...     def __repr__(self): return "SampleNode(%s)" % self.value
    ...
    >>> tree = SampleNode(0, SampleNode(1), [SampleNode(2), SampleNode(3)])
    >>> class MyVisitor(TreeVisitor):
    ...     def visit_SampleNode(self, node):
    ...         print("in %s %s" % (node.value, self.access_path))
    ...         self.visitchildren(node)
    ...         print("out %s" % node.value)
    ...
    >>> MyVisitor().visit(tree)
    in 0 []
    in 1 [(SampleNode(0), 'head', None)]
    out 1
    in 2 [(SampleNode(0), 'body', 0)]
    out 2
    in 3 [(SampleNode(0), 'body', 1)]
    out 3
    out 0
    """

    def __init__(self):
        if False:
            return 10
        super(TreeVisitor, self).__init__()
        self.dispatch_table = {}
        self.access_path = []

    def dump_node(self, node):
        if False:
            while True:
                i = 10
        ignored = list(node.child_attrs or []) + ['child_attrs', 'pos', 'gil_message', 'cpp_message', 'subexprs']
        values = []
        pos = getattr(node, 'pos', None)
        if pos:
            source = pos[0]
            if source:
                import os.path
                source = os.path.basename(source.get_description())
            values.append(u'%s:%s:%s' % (source, pos[1], pos[2]))
        attribute_names = dir(node)
        for attr in attribute_names:
            if attr in ignored:
                continue
            if attr.startswith('_') or attr.endswith('_'):
                continue
            try:
                value = getattr(node, attr)
            except AttributeError:
                continue
            if value is None or value == 0:
                continue
            elif isinstance(value, list):
                value = u'[...]/%d' % len(value)
            elif not isinstance(value, _PRINTABLE):
                continue
            else:
                value = repr(value)
            values.append(u'%s = %s' % (attr, value))
        return u'%s(%s)' % (node.__class__.__name__, u',\n    '.join(values))

    def _find_node_path(self, stacktrace):
        if False:
            print('Hello World!')
        import os.path
        last_traceback = stacktrace
        nodes = []
        while hasattr(stacktrace, 'tb_frame'):
            frame = stacktrace.tb_frame
            node = frame.f_locals.get('self')
            if isinstance(node, Nodes.Node):
                code = frame.f_code
                method_name = code.co_name
                pos = (os.path.basename(code.co_filename), frame.f_lineno)
                nodes.append((node, method_name, pos))
                last_traceback = stacktrace
            stacktrace = stacktrace.tb_next
        return (last_traceback, nodes)

    def _raise_compiler_error(self, child, e):
        if False:
            for i in range(10):
                print('nop')
        trace = ['']
        for (parent, attribute, index) in self.access_path:
            node = getattr(parent, attribute)
            if index is None:
                index = ''
            else:
                node = node[index]
                index = u'[%d]' % index
            trace.append(u'%s.%s%s = %s' % (parent.__class__.__name__, attribute, index, self.dump_node(node)))
        (stacktrace, called_nodes) = self._find_node_path(sys.exc_info()[2])
        last_node = child
        for (node, method_name, pos) in called_nodes:
            last_node = node
            trace.append(u"File '%s', line %d, in %s: %s" % (pos[0], pos[1], method_name, self.dump_node(node)))
        raise Errors.CompilerCrash(getattr(last_node, 'pos', None), self.__class__.__name__, u'\n'.join(trace), e, stacktrace)

    @cython.final
    def find_handler(self, obj):
        if False:
            while True:
                i = 10
        cls = type(obj)
        mro = inspect.getmro(cls)
        for mro_cls in mro:
            handler_method = getattr(self, 'visit_' + mro_cls.__name__, None)
            if handler_method is not None:
                return handler_method
        print(type(self), cls)
        if self.access_path:
            print(self.access_path)
            print(self.access_path[-1][0].pos)
            print(self.access_path[-1][0].__dict__)
        raise RuntimeError('Visitor %r does not accept object: %s' % (self, obj))

    def visit(self, obj):
        if False:
            print('Hello World!')
        return self._visit(obj)

    @cython.final
    def _visit(self, obj):
        if False:
            i = 10
            return i + 15
        try:
            try:
                handler_method = self.dispatch_table[type(obj)]
            except KeyError:
                handler_method = self.find_handler(obj)
                self.dispatch_table[type(obj)] = handler_method
            return handler_method(obj)
        except Errors.CompileError:
            raise
        except Errors.AbortError:
            raise
        except Exception as e:
            if DebugFlags.debug_no_exception_intercept:
                raise
            self._raise_compiler_error(obj, e)

    @cython.final
    def _visitchild(self, child, parent, attrname, idx):
        if False:
            i = 10
            return i + 15
        self.access_path.append((parent, attrname, idx))
        result = self._visit(child)
        self.access_path.pop()
        return result

    def visitchildren(self, parent, attrs=None, exclude=None):
        if False:
            return 10
        return self._visitchildren(parent, attrs, exclude)

    @cython.final
    @cython.locals(idx=cython.Py_ssize_t)
    def _visitchildren(self, parent, attrs, exclude):
        if False:
            i = 10
            return i + 15
        '\n        Visits the children of the given parent. If parent is None, returns\n        immediately (returning None).\n\n        The return value is a dictionary giving the results for each\n        child (mapping the attribute name to either the return value\n        or a list of return values (in the case of multiple children\n        in an attribute)).\n        '
        if parent is None:
            return None
        result = {}
        for attr in parent.child_attrs:
            if attrs is not None and attr not in attrs:
                continue
            if exclude is not None and attr in exclude:
                continue
            child = getattr(parent, attr)
            if child is not None:
                if type(child) is list:
                    childretval = [self._visitchild(x, parent, attr, idx) for (idx, x) in enumerate(child)]
                else:
                    childretval = self._visitchild(child, parent, attr, None)
                    assert not isinstance(childretval, list), 'Cannot insert list here: %s in %r' % (attr, parent)
                result[attr] = childretval
        return result

class VisitorTransform(TreeVisitor):
    """
    A tree transform is a base class for visitors that wants to do stream
    processing of the structure (rather than attributes etc.) of a tree.

    It implements __call__ to simply visit the argument node.

    It requires the visitor methods to return the nodes which should take
    the place of the visited node in the result tree (which can be the same
    or one or more replacement). Specifically, if the return value from
    a visitor method is:

    - [] or None; the visited node will be removed (set to None if an attribute and
    removed if in a list)
    - A single node; the visited node will be replaced by the returned node.
    - A list of nodes; the visited nodes will be replaced by all the nodes in the
    list. This will only work if the node was already a member of a list; if it
    was not, an exception will be raised. (Typically you want to ensure that you
    are within a StatListNode or similar before doing this.)
    """

    def visitchildren(self, parent, attrs=None, exclude=None):
        if False:
            for i in range(10):
                print('nop')
        return self._process_children(parent, attrs, exclude)

    @cython.final
    def _process_children(self, parent, attrs=None, exclude=None):
        if False:
            for i in range(10):
                print('nop')
        result = self._visitchildren(parent, attrs, exclude)
        for (attr, newnode) in result.items():
            if type(newnode) is list:
                newnode = self._flatten_list(newnode)
            setattr(parent, attr, newnode)
        return result

    @cython.final
    def _flatten_list(self, orig_list):
        if False:
            print('Hello World!')
        newlist = []
        for x in orig_list:
            if x is not None:
                if type(x) is list:
                    newlist.extend(x)
                else:
                    newlist.append(x)
        return newlist

    def visitchild(self, parent, attr, idx=0):
        if False:
            while True:
                i = 10
        child = getattr(parent, attr)
        if child is not None:
            node = self._visitchild(child, parent, attr, idx)
            if node is not child:
                setattr(parent, attr, node)
            child = node
        return child

    def recurse_to_children(self, node):
        if False:
            i = 10
            return i + 15
        self._process_children(node)
        return node

    def __call__(self, root):
        if False:
            return 10
        return self._visit(root)

class CythonTransform(VisitorTransform):
    """
    Certain common conventions and utilities for Cython transforms.

     - Sets up the context of the pipeline in self.context
     - Tracks directives in effect in self.current_directives
    """

    def __init__(self, context):
        if False:
            for i in range(10):
                print('nop')
        super(CythonTransform, self).__init__()
        self.context = context

    def __call__(self, node):
        if False:
            i = 10
            return i + 15
        from .ModuleNode import ModuleNode
        if isinstance(node, ModuleNode):
            self.current_directives = node.directives
        return super(CythonTransform, self).__call__(node)

    def visit_CompilerDirectivesNode(self, node):
        if False:
            return 10
        old = self.current_directives
        self.current_directives = node.directives
        self._process_children(node)
        self.current_directives = old
        return node

    def visit_Node(self, node):
        if False:
            print('Hello World!')
        self._process_children(node)
        return node

class ScopeTrackingTransform(CythonTransform):

    def visit_ModuleNode(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.scope_type = 'module'
        self.scope_node = node
        self._process_children(node)
        return node

    def visit_scope(self, node, scope_type):
        if False:
            print('Hello World!')
        prev = (self.scope_type, self.scope_node)
        self.scope_type = scope_type
        self.scope_node = node
        self._process_children(node)
        (self.scope_type, self.scope_node) = prev
        return node

    def visit_CClassDefNode(self, node):
        if False:
            print('Hello World!')
        return self.visit_scope(node, 'cclass')

    def visit_PyClassDefNode(self, node):
        if False:
            print('Hello World!')
        return self.visit_scope(node, 'pyclass')

    def visit_FuncDefNode(self, node):
        if False:
            while True:
                i = 10
        return self.visit_scope(node, 'function')

    def visit_CStructOrUnionDefNode(self, node):
        if False:
            return 10
        return self.visit_scope(node, 'struct')

class EnvTransform(CythonTransform):
    """
    This transformation keeps a stack of the environments.
    """

    def __call__(self, root):
        if False:
            while True:
                i = 10
        self.env_stack = []
        self.enter_scope(root, root.scope)
        return super(EnvTransform, self).__call__(root)

    def current_env(self):
        if False:
            i = 10
            return i + 15
        return self.env_stack[-1][1]

    def current_scope_node(self):
        if False:
            i = 10
            return i + 15
        return self.env_stack[-1][0]

    def global_scope(self):
        if False:
            for i in range(10):
                print('nop')
        return self.current_env().global_scope()

    def enter_scope(self, node, scope):
        if False:
            while True:
                i = 10
        self.env_stack.append((node, scope))

    def exit_scope(self):
        if False:
            print('Hello World!')
        self.env_stack.pop()

    def visit_FuncDefNode(self, node):
        if False:
            return 10
        self.visit_func_outer_attrs(node)
        self.enter_scope(node, node.local_scope)
        self.visitchildren(node, attrs=None, exclude=node.outer_attrs)
        self.exit_scope()
        return node

    def visit_func_outer_attrs(self, node):
        if False:
            print('Hello World!')
        self.visitchildren(node, attrs=node.outer_attrs)

    def visit_GeneratorBodyDefNode(self, node):
        if False:
            print('Hello World!')
        self._process_children(node)
        return node

    def visit_ClassDefNode(self, node):
        if False:
            return 10
        self.enter_scope(node, node.scope)
        self._process_children(node)
        self.exit_scope()
        return node

    def visit_CStructOrUnionDefNode(self, node):
        if False:
            print('Hello World!')
        self.enter_scope(node, node.scope)
        self._process_children(node)
        self.exit_scope()
        return node

    def visit_ScopedExprNode(self, node):
        if False:
            i = 10
            return i + 15
        if node.expr_scope:
            self.enter_scope(node, node.expr_scope)
            self._process_children(node)
            self.exit_scope()
        else:
            self._process_children(node)
        return node

    def visit_CArgDeclNode(self, node):
        if False:
            return 10
        if node.default:
            attrs = [attr for attr in node.child_attrs if attr != 'default']
            self._process_children(node, attrs)
            self.enter_scope(node, self.current_env().outer_scope)
            self.visitchildren(node, ('default',))
            self.exit_scope()
        else:
            self._process_children(node)
        return node

class NodeRefCleanupMixin(object):
    """
    Clean up references to nodes that were replaced.

    NOTE: this implementation assumes that the replacement is
    done first, before hitting any further references during
    normal tree traversal.  This needs to be arranged by calling
    "self.visitchildren()" at a proper place in the transform
    and by ordering the "child_attrs" of nodes appropriately.
    """

    def __init__(self, *args):
        if False:
            print('Hello World!')
        super(NodeRefCleanupMixin, self).__init__(*args)
        self._replacements = {}

    def visit_CloneNode(self, node):
        if False:
            for i in range(10):
                print('nop')
        arg = node.arg
        if arg not in self._replacements:
            self.visitchildren(arg)
        node.arg = self._replacements.get(arg, arg)
        return node

    def visit_ResultRefNode(self, node):
        if False:
            return 10
        expr = node.expression
        if expr is None or expr not in self._replacements:
            self.visitchildren(node)
            expr = node.expression
        if expr is not None:
            node.expression = self._replacements.get(expr, expr)
        return node

    def replace(self, node, replacement):
        if False:
            i = 10
            return i + 15
        self._replacements[node] = replacement
        return replacement
find_special_method_for_binary_operator = {'<': '__lt__', '<=': '__le__', '==': '__eq__', '!=': '__ne__', '>=': '__ge__', '>': '__gt__', '+': '__add__', '&': '__and__', '/': '__div__', '//': '__floordiv__', '<<': '__lshift__', '%': '__mod__', '*': '__mul__', '|': '__or__', '**': '__pow__', '>>': '__rshift__', '-': '__sub__', '^': '__xor__', 'in': '__contains__'}.get
find_special_method_for_unary_operator = {'not': '__not__', '~': '__inv__', '-': '__neg__', '+': '__pos__'}.get

class MethodDispatcherTransform(EnvTransform):
    """
    Base class for transformations that want to intercept on specific
    builtin functions or methods of builtin types, including special
    methods triggered by Python operators.  Must run after declaration
    analysis when entries were assigned.

    Naming pattern for handler methods is as follows:

    * builtin functions: _handle_(general|simple|any)_function_NAME

    * builtin methods: _handle_(general|simple|any)_method_TYPENAME_METHODNAME
    """

    def visit_GeneralCallNode(self, node):
        if False:
            print('Hello World!')
        self._process_children(node)
        function = node.function
        if not function.type.is_pyobject:
            return node
        arg_tuple = node.positional_args
        if not isinstance(arg_tuple, ExprNodes.TupleNode):
            return node
        keyword_args = node.keyword_args
        if keyword_args and (not isinstance(keyword_args, ExprNodes.DictNode)):
            return node
        args = arg_tuple.args
        return self._dispatch_to_handler(node, function, args, keyword_args)

    def visit_SimpleCallNode(self, node):
        if False:
            while True:
                i = 10
        self._process_children(node)
        function = node.function
        if function.type.is_pyobject:
            arg_tuple = node.arg_tuple
            if not isinstance(arg_tuple, ExprNodes.TupleNode):
                return node
            args = arg_tuple.args
        else:
            args = node.args
        return self._dispatch_to_handler(node, function, args, None)

    def visit_PrimaryCmpNode(self, node):
        if False:
            while True:
                i = 10
        if node.cascade:
            self._process_children(node)
            return node
        return self._visit_binop_node(node)

    def visit_BinopNode(self, node):
        if False:
            return 10
        return self._visit_binop_node(node)

    def _visit_binop_node(self, node):
        if False:
            for i in range(10):
                print('nop')
        self._process_children(node)
        special_method_name = find_special_method_for_binary_operator(node.operator)
        if special_method_name:
            (operand1, operand2) = (node.operand1, node.operand2)
            if special_method_name == '__contains__':
                (operand1, operand2) = (operand2, operand1)
            elif special_method_name == '__div__':
                if Future.division in self.current_env().global_scope().context.future_directives:
                    special_method_name = '__truediv__'
            obj_type = operand1.type
            if obj_type.is_builtin_type:
                type_name = obj_type.name
            else:
                type_name = 'object'
            node = self._dispatch_to_method_handler(special_method_name, None, False, type_name, node, None, [operand1, operand2], None)
        return node

    def visit_UnopNode(self, node):
        if False:
            for i in range(10):
                print('nop')
        self._process_children(node)
        special_method_name = find_special_method_for_unary_operator(node.operator)
        if special_method_name:
            operand = node.operand
            obj_type = operand.type
            if obj_type.is_builtin_type:
                type_name = obj_type.name
            else:
                type_name = 'object'
            node = self._dispatch_to_method_handler(special_method_name, None, False, type_name, node, None, [operand], None)
        return node

    def _find_handler(self, match_name, has_kwargs):
        if False:
            while True:
                i = 10
        try:
            match_name.encode('ascii')
        except UnicodeEncodeError:
            return None
        call_type = 'general' if has_kwargs else 'simple'
        handler = getattr(self, '_handle_%s_%s' % (call_type, match_name), None)
        if handler is None:
            handler = getattr(self, '_handle_any_%s' % match_name, None)
        return handler

    def _delegate_to_assigned_value(self, node, function, arg_list, kwargs):
        if False:
            return 10
        assignment = function.cf_state[0]
        value = assignment.rhs
        if value.is_name:
            if not value.entry or len(value.entry.cf_assignments) > 1:
                return node
        elif value.is_attribute and value.obj.is_name:
            if not value.obj.entry or len(value.obj.entry.cf_assignments) > 1:
                return node
        else:
            return node
        return self._dispatch_to_handler(node, value, arg_list, kwargs)

    def _dispatch_to_handler(self, node, function, arg_list, kwargs):
        if False:
            print('Hello World!')
        if function.is_name:
            if not function.entry:
                return node
            entry = function.entry
            is_builtin = entry.is_builtin or entry is self.current_env().builtin_scope().lookup_here(function.name)
            if not is_builtin:
                if function.cf_state and function.cf_state.is_single:
                    return self._delegate_to_assigned_value(node, function, arg_list, kwargs)
                if arg_list and entry.is_cmethod and entry.scope and entry.scope.parent_type.is_builtin_type:
                    if entry.scope.parent_type is arg_list[0].type:
                        return self._dispatch_to_method_handler(entry.name, self_arg=None, is_unbound_method=True, type_name=entry.scope.parent_type.name, node=node, function=function, arg_list=arg_list, kwargs=kwargs)
                return node
            function_handler = self._find_handler('function_%s' % function.name, kwargs)
            if function_handler is None:
                return self._handle_function(node, function.name, function, arg_list, kwargs)
            if kwargs:
                return function_handler(node, function, arg_list, kwargs)
            else:
                return function_handler(node, function, arg_list)
        elif function.is_attribute:
            attr_name = function.attribute
            if function.type.is_pyobject:
                self_arg = function.obj
            elif node.self and function.entry:
                entry = function.entry.as_variable
                if not entry or not entry.is_builtin:
                    return node
                self_arg = node.self
                arg_list = arg_list[1:]
            else:
                return node
            obj_type = self_arg.type
            is_unbound_method = False
            if obj_type.is_builtin_type:
                if obj_type is Builtin.type_type and self_arg.is_name and arg_list and arg_list[0].type.is_pyobject:
                    type_name = self_arg.name
                    self_arg = None
                    is_unbound_method = True
                else:
                    type_name = obj_type.name
            else:
                type_name = 'object'
            return self._dispatch_to_method_handler(attr_name, self_arg, is_unbound_method, type_name, node, function, arg_list, kwargs)
        else:
            return node

    def _dispatch_to_method_handler(self, attr_name, self_arg, is_unbound_method, type_name, node, function, arg_list, kwargs):
        if False:
            while True:
                i = 10
        method_handler = self._find_handler('method_%s_%s' % (type_name, attr_name), kwargs)
        if method_handler is None:
            if attr_name in TypeSlots.special_method_names or attr_name in ['__new__', '__class__']:
                method_handler = self._find_handler('slot%s' % attr_name, kwargs)
            if method_handler is None:
                return self._handle_method(node, type_name, attr_name, function, arg_list, is_unbound_method, kwargs)
        if self_arg is not None:
            arg_list = [self_arg] + list(arg_list)
        if kwargs:
            result = method_handler(node, function, arg_list, is_unbound_method, kwargs)
        else:
            result = method_handler(node, function, arg_list, is_unbound_method)
        return result

    def _handle_function(self, node, function_name, function, arg_list, kwargs):
        if False:
            i = 10
            return i + 15
        'Fallback handler'
        return node

    def _handle_method(self, node, type_name, attr_name, function, arg_list, is_unbound_method, kwargs):
        if False:
            return 10
        'Fallback handler'
        return node

class RecursiveNodeReplacer(VisitorTransform):
    """
    Recursively replace all occurrences of a node in a subtree by
    another node.
    """

    def __init__(self, orig_node, new_node):
        if False:
            i = 10
            return i + 15
        super(RecursiveNodeReplacer, self).__init__()
        (self.orig_node, self.new_node) = (orig_node, new_node)

    def visit_CloneNode(self, node):
        if False:
            print('Hello World!')
        if node is self.orig_node:
            return self.new_node
        if node.arg is self.orig_node:
            node.arg = self.new_node
        return node

    def visit_Node(self, node):
        if False:
            print('Hello World!')
        self._process_children(node)
        if node is self.orig_node:
            return self.new_node
        else:
            return node

def recursively_replace_node(tree, old_node, new_node):
    if False:
        print('Hello World!')
    replace_in = RecursiveNodeReplacer(old_node, new_node)
    replace_in(tree)

class NodeFinder(TreeVisitor):
    """
    Find out if a node appears in a subtree.
    """

    def __init__(self, node):
        if False:
            return 10
        super(NodeFinder, self).__init__()
        self.node = node
        self.found = False

    def visit_Node(self, node):
        if False:
            return 10
        if self.found:
            pass
        elif node is self.node:
            self.found = True
        else:
            self._visitchildren(node, None, None)

def tree_contains(tree, node):
    if False:
        return 10
    finder = NodeFinder(node)
    finder.visit(tree)
    return finder.found

def replace_node(ptr, value):
    if False:
        return 10
    'Replaces a node. ptr is of the form used on the access path stack\n    (parent, attrname, listidx|None)\n    '
    (parent, attrname, listidx) = ptr
    if listidx is None:
        setattr(parent, attrname, value)
    else:
        getattr(parent, attrname)[listidx] = value

class PrintTree(TreeVisitor):
    """Prints a representation of the tree to standard output.
    Subclass and override repr_of to provide more information
    about nodes. """

    def __init__(self, start=None, end=None):
        if False:
            return 10
        TreeVisitor.__init__(self)
        self._indent = ''
        if start is not None or end is not None:
            self._line_range = (start or 0, end or 2 ** 30)
        else:
            self._line_range = None

    def indent(self):
        if False:
            return 10
        self._indent += '  '

    def unindent(self):
        if False:
            print('Hello World!')
        self._indent = self._indent[:-2]

    def __call__(self, tree, phase=None):
        if False:
            for i in range(10):
                print('nop')
        print("Parse tree dump at phase '%s'" % phase)
        self.visit(tree)
        return tree

    def visit_Node(self, node):
        if False:
            print('Hello World!')
        self._print_node(node)
        self.indent()
        self.visitchildren(node)
        self.unindent()
        return node

    def visit_CloneNode(self, node):
        if False:
            return 10
        self._print_node(node)
        self.indent()
        line = node.pos[1]
        if self._line_range is None or self._line_range[0] <= line <= self._line_range[1]:
            print('%s- %s: %s' % (self._indent, 'arg', self.repr_of(node.arg)))
        self.indent()
        self.visitchildren(node.arg)
        self.unindent()
        self.unindent()
        return node

    def _print_node(self, node):
        if False:
            i = 10
            return i + 15
        line = node.pos[1]
        if self._line_range is None or self._line_range[0] <= line <= self._line_range[1]:
            if len(self.access_path) == 0:
                name = '(root)'
            else:
                (parent, attr, idx) = self.access_path[-1]
                if idx is not None:
                    name = '%s[%d]' % (attr, idx)
                else:
                    name = attr
            print('%s- %s: %s' % (self._indent, name, self.repr_of(node)))

    def repr_of(self, node):
        if False:
            print('Hello World!')
        if node is None:
            return '(none)'
        else:
            result = node.__class__.__name__
            if isinstance(node, ExprNodes.NameNode):
                result += '(type=%s, name="%s")' % (repr(node.type), node.name)
            elif isinstance(node, Nodes.DefNode):
                result += '(name="%s")' % node.name
            elif isinstance(node, Nodes.CFuncDefNode):
                result += '(name="%s")' % node.declared_name()
            elif isinstance(node, ExprNodes.AttributeNode):
                result += '(type=%s, attribute="%s")' % (repr(node.type), node.attribute)
            elif isinstance(node, (ExprNodes.ConstNode, ExprNodes.PyConstNode)):
                result += '(type=%s, value=%r)' % (repr(node.type), node.value)
            elif isinstance(node, ExprNodes.ExprNode):
                t = node.type
                result += '(type=%s)' % repr(t)
            elif node.pos:
                pos = node.pos
                path = pos[0].get_description()
                if '/' in path:
                    path = path.split('/')[-1]
                if '\\' in path:
                    path = path.split('\\')[-1]
                result += '(pos=(%s:%s:%s))' % (path, pos[1], pos[2])
            return result
if __name__ == '__main__':
    import doctest
    doctest.testmod()