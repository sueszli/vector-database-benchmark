import contextlib
import json
import re
import os
from os import path
from io import StringIO
from jinja2 import Environment, FileSystemLoader, nodes
from pathlib import Path
import srsly
import sys
OPERANDS = {'eq': '===', 'ne': '!==', 'lt': ' < ', 'gt': ' > ', 'lteq': ' <= ', 'gteq': ' >= '}
DICT_ITER_METHODS = ('iteritems', 'items', 'values', 'keys')
STATE_DEFAULT = 0
STATE_EXECUTING = 1
STATE_INTERPOLATING = 2
LOOP_HELPER_INDEX = 'index'
LOOP_HELPER_INDEX_0 = 'index0'
LOOP_HELPER_FIRST = 'first'
LOOP_HELPER_LAST = 'last'
LOOP_HELPER_LENGTH = 'length'
LOOP_HELPERS = (LOOP_HELPER_INDEX, LOOP_HELPER_INDEX_0, LOOP_HELPER_FIRST, LOOP_HELPER_LAST, LOOP_HELPER_LENGTH)

def amd_format(dependencies, template_function):
    if False:
        print('Hello World!')
    result = 'define(['
    result += ','.join(('"{0}"'.format(x[0]) for x in dependencies))
    result += '], function ('
    result += ','.join((x[1] for x in dependencies))
    result += ') { return '
    result += template_function
    result += '; });'
    return result

def commonjs_format(dependencies, template_function):
    if False:
        while True:
            i = 10
    result = ''.join(('var {0} = require("{1}");'.format(y, x) for (x, y) in dependencies))
    result += 'module.exports = {0};'.format(template_function)
    return result

def es6_format(dependencies, template_function):
    if False:
        i = 10
        return i + 15
    result = ''.join(('import {0} from "{1}";'.format(y, x) for (x, y) in dependencies))
    result += 'export default {0}'.format(template_function)
    return result
JS_MODULE_FORMATS = {None: lambda dependencies, template_function: template_function, 'amd': amd_format, 'commonjs': commonjs_format, 'es6': es6_format}
TEMPLATE_WRAPPER = '\nfunction {function_name}(ctx) {{\n    var __result = "";\n    var __tmp;\n    var __runtime = jinjaToJS.runtime;\n    var __filters = jinjaToJS.filters;\n    var __globals = jinjaToJS.globals;\n    var context = jinjaToJS.createContext(ctx);\n    {template_code}\n    return __result;\n}}\n'

class ExtendsException(Exception):
    """
    Raised when an {% extends %} is encountered. At this point the parent template is
    loaded and all blocks defined in the current template passed to it.
    """
    pass

@contextlib.contextmanager
def option(current_kwargs, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Context manager for temporarily setting a keyword argument and\n    then restoring it to whatever it was before.\n    '
    tmp_kwargs = dict(((key, current_kwargs.get(key)) for (key, value) in kwargs.items()))
    current_kwargs.update(kwargs)
    yield
    current_kwargs.update(tmp_kwargs)

def is_method_call(node, method_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns True if `node` is a method call for `method_name`. `method_name`\n    can be either a string or an iterable of strings.\n    '
    if not isinstance(node, nodes.Call):
        return False
    if isinstance(node.node, nodes.Getattr):
        method = node.node.attr
    elif isinstance(node.node, nodes.Name):
        method = node.node.name
    elif isinstance(node.node, nodes.Getitem):
        method = node.node.arg.value
    else:
        return False
    if isinstance(method_name, (list, tuple)):
        return method in method_name
    return method == method_name

def is_loop_helper(node):
    if False:
        i = 10
        return i + 15
    '\n    Returns True is node is a loop helper e.g. {{ loop.index }} or {{ loop.first }}\n    '
    return hasattr(node, 'node') and isinstance(node.node, nodes.Name) and (node.node.name == 'loop')

def temp_var_names_generator():
    if False:
        i = 10
        return i + 15
    x = 0
    while True:
        yield ('__$%s' % x)
        x += 1

class JinjaToJS(object):

    def __init__(self, template_root, template_name, js_module_format=None, runtime_path='jinja-to-js', include_prefix='', include_ext='', child_blocks=None, dependencies=None, custom_filters=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Args:\n            template_root (str): The path to where templates should be loaded from.\n            template_name (str): The name of the template to compile (relative to `template_root`).\n            js_module_format (str, optional): The JavaScript module format to use.\n                                              One of ('amd', 'commonjs', 'es6')\n            runtime_path (str, optional): If `js_module_format` is specified then the JavaScript\n                                          runtime will be imported using the appropriate method.\n                                          It defaults to assuming it will be imported from\n                                          `node_modules` but you can change it using this option.\n            include_prefix (str, optional): If using the `amd` module format you can use this option\n                                            to add a prefix to every include path as AMD imports are\n                                            generally relative to the main file, not the module\n                                            importing.\n            include_ext (str, optional): By default any includes will be references without an\n                                         extension, as neither AMD, commonJS or ES6 require the\n                                         '.js' extension. If you want to use an extension, say\n                                         '.template' then set this option to a string including\n                                         the leading '.'\n            child_blocks (dict, optional): Used internally when handling templates that extend\n                                           other templates.\n            dependencies (list of tuple, optional): Used internally when handling templates that\n                                                    extend other templates.\n            custom_filters (list of str, optional): List of custom filters which should be allowed.\n                                                    These may be filters supported by Jinja but not\n                                                    supported by jinja-to-js. These filters MUST be\n                                                    registered with the jinja-to-js JS runtime.\n        "
        self.environment = Environment(loader=FileSystemLoader(template_root), autoescape=True)
        self.output = StringIO()
        self.stored_names = set()
        self.temp_var_names = temp_var_names_generator()
        self.state = STATE_DEFAULT
        self.child_blocks = child_blocks or {}
        self.dependencies = dependencies or []
        self._runtime_function_cache = []
        self.js_module_format = js_module_format
        self.runtime_path = runtime_path
        self.include_prefix = include_prefix
        self.include_ext = include_ext
        self.template_root = template_root
        self.template_name = template_name
        self.custom_filters = custom_filters or []
        self.js_function_name = 'template' + ''.join((x.title() for x in re.split('[^\\w]|_', path.splitext(self.template_name)[0])))
        self.context_name = 'context'
        self._add_dependency(self.runtime_path, 'jinjaToJS')
        if os.name == 'nt':
            self.template_name = self.template_name.replace(os.pathsep, '/')
        (template_string, template_path, _) = self.environment.loader.get_source(self.environment, self.template_name)
        self.template_path = template_path
        if self.js_module_format not in JS_MODULE_FORMATS.keys():
            raise ValueError('The js_module_format option must be one of: %s' % JS_MODULE_FORMATS.keys())
        self.ast = self.environment.parse(template_string)
        try:
            for node in self.ast.body:
                self._process_node(node)
        except ExtendsException:
            pass

    def get_output(self):
        if False:
            return 10
        '\n        Returns the generated JavaScript code.\n\n        Returns:\n            str\n        '
        template_function = TEMPLATE_WRAPPER.format(function_name=self.js_function_name, template_code=self.output.getvalue()).strip()
        module_format = JS_MODULE_FORMATS[self.js_module_format]
        return module_format(self.dependencies, template_function)

    def _get_depencency_var_name(self, dependency):
        if False:
            return 10
        '\n        Returns the variable name assigned to the given dependency or None if the dependency has\n        not yet been registered.\n\n        Args:\n            dependency (str): Thet dependency that needs to be imported.\n\n        Returns:\n            str or None\n        '
        for (dep_path, var_name) in self.dependencies:
            if dep_path == dependency:
                return var_name

    def _add_dependency(self, dependency, var_name=None):
        if False:
            i = 10
            return i + 15
        '\n        Adds the given dependency and returns the variable name to use to access it. If `var_name`\n        is not given then a random one will be created.\n\n        Args:\n            dependency (str):\n            var_name (str, optional):\n\n        Returns:\n            str\n        '
        if var_name is None:
            var_name = next(self.temp_var_names)
        if (dependency, var_name) not in self.dependencies:
            self.dependencies.append((dependency, var_name))
        return var_name

    def _process_node(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        node_name = node.__class__.__name__.lower()
        handler = getattr(self, '_process_' + node_name, None)
        if callable(handler):
            handler(node, **kwargs)
        else:
            raise Exception(f'Unknown node {node} ({node_name})')

    def _process_extends(self, node, **kwargs):
        if False:
            print('Hello World!')
        '\n        Processes an extends block e.g. `{% extends "some/template.jinja" %}`\n        '
        for b in self.ast.find_all(nodes.Block):
            if b.name not in self.child_blocks:
                self.child_blocks[b.name] = b
            else:
                block = self.child_blocks.get(b.name)
                while hasattr(block, 'super_block'):
                    block = block.super_block
                block.super_block = b
        parent_template = JinjaToJS(template_root=self.template_root, template_name=node.template.value, js_module_format=self.js_module_format, runtime_path=self.runtime_path, include_prefix=self.include_prefix, include_ext=self.include_ext, child_blocks=self.child_blocks, dependencies=self.dependencies)
        self.output.write(parent_template.output.getvalue())
        raise ExtendsException

    def _process_block(self, node, **kwargs):
        if False:
            print('Hello World!')
        '\n        Processes a block e.g. `{% block my_block %}{% endblock %}`\n        '
        if not hasattr(node, 'super_block'):
            node.super_block = None
            child_block = self.child_blocks.get(node.name)
            if child_block:
                last_block = child_block
                while hasattr(last_block, 'super_block'):
                    last_block = child_block.super_block
                last_block.super_block = node
                node = child_block
        for n in node.body:
            self._process_node(n, super_block=node.super_block, **kwargs)

    def _process_output(self, node, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Processes an output node, which will contain things like `Name` and `TemplateData` nodes.\n        '
        for n in node.nodes:
            self._process_node(n, **kwargs)

    def _process_templatedata(self, node, **_):
        if False:
            i = 10
            return i + 15
        '\n        Processes a `TemplateData` node, this is just a bit of as-is text\n        to be written to the output.\n        '
        value = re.sub('"', '\\\\"', node.data)
        value = re.sub('\n', '\\\\n', value)
        self.output.write('__result += "' + value + '";')

    def _process_name(self, node, **kwargs):
        if False:
            print('Hello World!')
        "\n        Processes a `Name` node. Some examples of `Name` nodes:\n            {{ foo }} -> 'foo' is a Name\n            {% if foo }} -> 'foo' is a Name\n        "
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs):
                if node.name not in self.stored_names and node.ctx != 'store':
                    self.output.write(self.context_name)
                    self.output.write('.')
                if node.ctx == 'store':
                    self.stored_names.add(node.name)
                self.output.write(node.name)

    def _process_dict(self, node, **kwargs):
        if False:
            print('Hello World!')
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs):
                if node.items:
                    err = f"Can't process non-empty dict in expression: {node}"
                    raise ValueError(err)
                self.output.write('{}')

    def _process_getattr(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Processes a `GetAttr` node. e.g. {{ foo.bar }}\n        '
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                if is_loop_helper(node):
                    self._process_loop_helper(node, **new_kwargs)
                else:
                    self._process_node(node.node, **new_kwargs)
                    self.output.write('.')
                    self.output.write(node.attr)

    def _process_getitem(self, node, **kwargs):
        if False:
            print('Hello World!')
        '\n        Processes a `GetItem` node e.g. {{ foo["bar"] }}\n        '
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self._process_node(node.node, **new_kwargs)
                if isinstance(node.arg, nodes.Slice):
                    self.output.write('.slice(')
                    if node.arg.step is not None:
                        raise Exception('The step argument is not supported when slicing.')
                    if node.arg.start is None:
                        self.output.write('0')
                    else:
                        self._process_node(node.arg.start, **new_kwargs)
                    if node.arg.stop is None:
                        self.output.write(')')
                    else:
                        self.output.write(',')
                        self._process_node(node.arg.stop, **new_kwargs)
                        self.output.write(')')
                else:
                    self.output.write('[')
                    self._process_node(node.arg, **new_kwargs)
                    self.output.write(']')

    def _process_for(self, node, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Processes a for loop. e.g.\n            {% for number in numbers %}\n                {{ number }}\n            {% endfor %}\n            {% for key, value in somemap.items() %}\n                {{ key }} -> {{ value }}\n            {% %}\n        '
        previous_stored_names = self.stored_names.copy()
        with self._execution():
            self.output.write('__runtime.each(')
            if is_method_call(node.iter, dict.keys.__name__):
                self.output.write('Object.keys(')
            self._process_node(node.iter, **kwargs)
            if is_method_call(node.iter, dict.keys.__name__):
                self.output.write(')')
            self.output.write(',')
            self.output.write('function')
            self.output.write('(')
            if isinstance(node.target, nodes.Tuple):
                if len(node.target.items) > 2:
                    raise Exception('De-structuring more than 2 items is not supported.')
                for (i, item) in enumerate(reversed(node.target.items)):
                    self._process_node(item, **kwargs)
                    if i < len(node.target.items) - 1:
                        self.output.write(',')
            else:
                self._process_node(node.target, **kwargs)
            self.output.write(')')
            self.output.write('{')
            if node.test:
                self.output.write('if (!(')
                self._process_node(node.test, **kwargs)
                self.output.write(')) { return; }')
        assigns = node.target.items if isinstance(node.target, nodes.Tuple) else [node.target]
        with self._scoped_variables(assigns, **kwargs):
            for n in node.body:
                self._process_node(n, **kwargs)
        with self._execution():
            self.output.write('}')
            self.output.write(')')
            self.output.write(';')
        self.stored_names = previous_stored_names

    def _process_if(self, node, execute_end=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Processes an if block e.g. `{% if foo %} do something {% endif %}`\n        '
        with self._execution():
            self.output.write('if')
            self.output.write('(')
            with option(kwargs, use_python_bool_wrapper=True):
                self._process_node(node.test, **kwargs)
            self.output.write(')')
            self.output.write('{')
        if execute_end:
            execute_end()
        for n in node.body:
            self._process_node(n, **kwargs)
        if not node.else_ and (not node.elif_):
            with self._execution():
                self.output.write('}')
        else:
            with self._execution() as execute_end:
                self.output.write('}')
                self.output.write(' else ')
                for n in node.elif_:
                    self._process_node(n, execute_end=execute_end, **kwargs)
                if node.elif_ and node.else_:
                    self.output.write(' else ')
                self.output.write('{')
            for n in node.else_:
                self._process_node(n, **kwargs)
            with self._execution():
                self.output.write('}')

    def _process_condexpr(self, node, **kwargs):
        if False:
            return 10
        with self._interpolation():
            self.output.write('(')
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self._process_node(node.test, **new_kwargs)
            self.output.write(' ? ')
            self._process_node(node.expr1, **kwargs)
            self.output.write(' : ')
            self._process_node(node.expr2, **kwargs)
            self.output.write(')')

    def _process_not(self, node, **kwargs):
        if False:
            i = 10
            return i + 15
        self.output.write('!')
        with self._python_bool_wrapper(**kwargs) as new_kwargs:
            self._process_node(node.node, **new_kwargs)

    def _process_or(self, node, **kwargs):
        if False:
            return 10
        self._process_node(node.left, **kwargs)
        self.output.write(' || ')
        self._process_node(node.right, **kwargs)

    def _process_and(self, node, **kwargs):
        if False:
            return 10
        self._process_node(node.left, **kwargs)
        self.output.write(' && ')
        self._process_node(node.right, **kwargs)

    def _process_tuple(self, node, **kwargs):
        if False:
            while True:
                i = 10
        self.output.write('[')
        for (i, item) in enumerate(node.items):
            self._process_node(item, **kwargs)
            if i < len(node.items) - 1:
                self.output.write(',')
        self.output.write(']')

    def _process_call(self, node, super_block=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if is_method_call(node, DICT_ITER_METHODS):
            self._process_node(node.node.node, **kwargs)
        elif is_method_call(node, 'super'):
            if not super_block:
                raise Exception('super() called outside of a block with a parent.')
            self._process_node(super_block, **kwargs)
        else:
            with self._interpolation():
                with self._python_bool_wrapper(**kwargs) as new_kwargs:
                    self._process_node(node.node, **new_kwargs)
                    self.output.write('(')
                    self._process_args(node, **new_kwargs)
                    self.output.write(')')
                    if self.state != STATE_INTERPOLATING:
                        self.output.write('')

    def _process_filter(self, node, **kwargs):
        if False:
            return 10
        method_name = getattr(self, '_process_filter_%s' % node.name, None)
        if callable(method_name):
            method_name(node, **kwargs)
        elif node.name in self.custom_filters:
            with self._interpolation(safe=True):
                with self._python_bool_wrapper(**kwargs) as new_kwargs:
                    self.output.write('__filters.%s(' % node.name)
                    self._process_node(node.node, **new_kwargs)
                    if getattr(node, 'args', None):
                        self.output.write(',')
                        self._process_args(node, **new_kwargs)
                    self.output.write(')')
        else:
            raise Exception('Unsupported filter: %s' % node.name)

    def _process_filter_safe(self, node, **kwargs):
        if False:
            return 10
        with self._interpolation(safe=True):
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self._process_node(node.node, **new_kwargs)

    def _process_filter_capitalize(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.capitalize(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_abs(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('Math.abs(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_replace(self, node, **kwargs):
        if False:
            print('Hello World!')
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self._process_node(node.node, **new_kwargs)
                self.output.write('.split(')
                self._process_node(node.args[0], **new_kwargs)
                self.output.write(').join(')
                self._process_node(node.args[1], **new_kwargs)
                self.output.write(')')

    def _process_filter_pprint(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('JSON.stringify(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_attr(self, node, **kwargs):
        if False:
            print('Hello World!')
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self._process_node(node.node, **new_kwargs)
                self.output.write('[')
                self._process_node(node.args[0], **new_kwargs)
                self.output.write(']')

    def _process_filter_batch(self, node, **kwargs):
        if False:
            while True:
                i = 10
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.batch(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(',')
                self._process_args(node, **new_kwargs)
                self.output.write(')')

    def _process_filter_default(self, node, **kwargs):
        if False:
            while True:
                i = 10
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.default(')
                self._process_node(node.node, **new_kwargs)
                if node.args:
                    self.output.write(',')
                self._process_args(node, **new_kwargs)
                self.output.write(')')

    def _process_filter_first(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.first(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_int(self, node, **kwargs):
        if False:
            print('Hello World!')
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.int(')
                self._process_node(node.node, **new_kwargs)
                if node.args:
                    self.output.write(',')
                self._process_args(node, **new_kwargs)
                self.output.write(')')

    def _process_filter_round(self, node, **kwargs):
        if False:
            while True:
                i = 10
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('Math.round((')
                self._process_node(node.node, **new_kwargs)
                self.output.write('+ Number.EPSILON) * 10**')
                self._process_node(node.args[0], **new_kwargs)
                self.output.write(') / 10**')
                self._process_node(node.args[0], **new_kwargs)

    def _process_filter_last(self, node, **kwargs):
        if False:
            return 10
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.last(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_length(self, node, **kwargs):
        if False:
            i = 10
            return i + 15
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.size(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_lower(self, node, **kwargs):
        if False:
            while True:
                i = 10
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(' + "").toLowerCase()')

    def _process_filter_slice(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.slice(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(',')
                self._process_args(node, **new_kwargs)
                self.output.write(')')

    def _process_filter_title(self, node, **kwargs):
        if False:
            return 10
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.title(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_trim(self, node, **kwargs):
        if False:
            i = 10
            return i + 15
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(' + "").trim()')

    def _process_filter_upper(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(' + "").toUpperCase()')

    def _process_filter_truncate(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.truncate(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(',')
                self._process_args(node, **new_kwargs)
                self.output.write(')')

    def _process_assign(self, node, **kwargs):
        if False:
            return 10
        with self._execution():
            self.output.write('var ')
            self._process_node(node.target, **kwargs)
            self.output.write(' = ')
            self._process_node(node.node, **kwargs)
            self.output.write(';')

    def _process_with(self, node, **kwargs):
        if False:
            print('Hello World!')
        previous_stored_names = self.stored_names.copy()
        assigns_in_tag = [nodes.Assign(t, v) for (t, v) in zip(node.targets, node.values)]
        assigns_in_body = [x for x in node.body if isinstance(x, nodes.Assign)]
        node.body = [x for x in node.body if not isinstance(x, nodes.Assign)]
        all_assigns = assigns_in_tag + assigns_in_body
        with self._execution():
            self.output.write('(function () {')
        with self._scoped_variables(all_assigns, **kwargs):
            for node in node.body:
                self._process_node(node, **kwargs)
        with self._execution():
            self.output.write('})();')
        self.stored_names = previous_stored_names

    def _process_compare(self, node, **kwargs):
        if False:
            return 10
        if len(node.ops) > 1:
            raise Exception('Multiple operands are not supported.')
        operand = node.ops[0]
        is_equality = operand.op in ('eq', 'ne')
        left_hand_is_const = isinstance(node.expr, nodes.Const)
        right_hand_is_const = isinstance(operand.expr, nodes.Const)
        use_is_equal_function = is_equality and (not (left_hand_is_const or right_hand_is_const))
        with option(kwargs, use_python_bool_wrapper=False):
            if operand.op == 'in' or operand.op == 'notin':
                if operand.op == 'notin':
                    self.output.write('!')
                self._process_node(operand.expr, **kwargs)
                self.output.write('.includes(')
                self._process_node(node.expr, **kwargs)
                self.output.write(')')
            else:
                if use_is_equal_function:
                    if operand.op == 'ne':
                        self.output.write('!')
                    self.output.write('__runtime.isEqual(')
                self._process_node(node.expr, **kwargs)
                if use_is_equal_function:
                    self.output.write(',')
                else:
                    self.output.write(OPERANDS.get(operand.op))
                self._process_node(operand.expr, **kwargs)
                if use_is_equal_function:
                    self.output.write(')')

    def _process_operand(self, node, **kwargs):
        if False:
            while True:
                i = 10
        self.output.write(OPERANDS.get(node.op))
        self._process_node(node.expr, **kwargs)

    def _process_const(self, node, **_):
        if False:
            return 10
        with self._interpolation():
            self.output.write(json.dumps(node.value))

    def _process_nonetype(self, node, **_):
        if False:
            i = 10
            return i + 15
        with self._interpolation():
            self.output.write('null')

    def _process_neg(self, node, **kwargs):
        if False:
            i = 10
            return i + 15
        with self._interpolation():
            self.output.write('-')
            self._process_node(node.node, **kwargs)

    def _process_list(self, node, **kwargs):
        if False:
            return 10
        self.output.write('[')
        for (i, item) in enumerate(node.items):
            self._process_node(item, **kwargs)
            if i < len(node.items) - 1:
                self.output.write(',')
        self.output.write(']')

    def _process_test(self, node, **kwargs):
        if False:
            while True:
                i = 10
        with option(kwargs, use_python_bool_wrapper=False):
            method_name = getattr(self, '_process_test_%s' % node.name, None)
            if callable(method_name):
                method_name(node, **kwargs)
            else:
                raise Exception('Unsupported test: %s' % node.name)

    def _process_test_defined(self, node, **kwargs):
        if False:
            i = 10
            return i + 15
        self.output.write('(typeof ')
        self._process_node(node.node, **kwargs)
        self.output.write(' !== "undefined")')

    def _process_test_undefined(self, node, **kwargs):
        if False:
            print('Hello World!')
        self._process_node(node.node, **kwargs)
        self.output.write(' === undefined')

    def _process_test_callable(self, node, **kwargs):
        if False:
            while True:
                i = 10
        self.output.write('__runtime.type(')
        self._process_node(node.node, **kwargs)
        self.output.write(') === "Function"')

    def _process_test_divisibleby(self, node, **kwargs):
        if False:
            return 10
        self._process_node(node.node, **kwargs)
        self.output.write(' % ')
        self._process_node(node.args[0], **kwargs)
        self.output.write(' === 0')

    def _process_test_even(self, node, **kwargs):
        if False:
            while True:
                i = 10
        self._process_node(node.node, **kwargs)
        self.output.write(' % 2 === 0')

    def _process_test_odd(self, node, **kwargs):
        if False:
            i = 10
            return i + 15
        self._process_node(node.node, **kwargs)
        self.output.write(' % 2 === 1')

    def _process_test_none(self, node, **kwargs):
        if False:
            return 10
        self._process_node(node.node, **kwargs)
        self.output.write(' === null')

    def _process_test_upper(self, node, **kwargs):
        if False:
            i = 10
            return i + 15
        self._process_node(node.node, **kwargs)
        self.output.write('.toUpperCase() === ')
        self._process_node(node.node, **kwargs)

    def _process_test_lower(self, node, **kwargs):
        if False:
            print('Hello World!')
        self._process_node(node.node, **kwargs)
        self.output.write('.toLowerCase() === ')
        self._process_node(node.node, **kwargs)

    def _process_test_string(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.output.write('__runtime.type(')
        self._process_node(node.node, **kwargs)
        self.output.write(') === "String"')

    def _process_test_mapping(self, node, **kwargs):
        if False:
            i = 10
            return i + 15
        self.output.write('__runtime.type(')
        self._process_node(node.node, **kwargs)
        self.output.write(') === "Object"')

    def _process_test_number(self, node, **kwargs):
        if False:
            return 10
        self.output.write('(__runtime.type(')
        self._process_node(node.node, **kwargs)
        self.output.write(') === "Number" && !isNaN(')
        self._process_node(node.node, **kwargs)
        self.output.write('))')

    def _process_include(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with self._interpolation(safe=True):
            include_path = node.template.value
            if include_path == self.template_name:
                include_var_name = self.js_function_name
            else:
                if self.include_prefix:
                    include_path = self.include_prefix + node.template.value
                elif self.js_module_format in ('es6', 'commonjs') and self.template_name:
                    (_, absolute_include_path, _) = self.environment.loader.get_source(self.environment, node.template.value)
                    include_path = os.path.relpath(absolute_include_path, os.path.dirname(self.template_path))
                    if not include_path.startswith('.'):
                        include_path = './' + include_path
                if os.name == 'nt':
                    include_path = include_path.replace(os.pathsep, '/')
                include_path = path.splitext(include_path)[0] + self.include_ext
                include_var_name = self._get_depencency_var_name(include_path)
                if not include_var_name:
                    include_var_name = self._add_dependency(include_path)
            if self.js_module_format is None:
                self.output.write('jinjaToJS.include("')
                self.output.write(include_path)
                self.output.write('");')
            else:
                self.output.write(include_var_name)
            self.output.write('(')
            self.output.write(self.context_name)
            self.output.write(')')

    def _process_add(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node.left, nodes.List) or isinstance(node.right, nodes.List):
            with self._interpolation():
                with self._python_bool_wrapper(**kwargs) as new_kwargs:
                    self._process_node(node.left, **new_kwargs)
                    self.output.write('.concat(')
                    self._process_node(node.right, **new_kwargs)
                    self.output.write(')')
        else:
            self._process_math(node, math_operator=' + ', **kwargs)

    def _process_sub(self, node, **kwargs):
        if False:
            return 10
        self._process_math(node, math_operator=' - ', **kwargs)

    def _process_div(self, node, **kwargs):
        if False:
            while True:
                i = 10
        self._process_math(node, math_operator=' / ', **kwargs)

    def _process_floordiv(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._process_math(node, math_operator=' / ', function='Math.floor', **kwargs)

    def _process_mul(self, node, **kwargs):
        if False:
            i = 10
            return i + 15
        self._process_math(node, math_operator=' * ', **kwargs)

    def _process_mod(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._process_math(node, math_operator=' % ', **kwargs)

    def _process_math(self, node, math_operator=None, function=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Processes a math node e.g. `Div`, `Sub`, `Add`, `Mul` etc...\n        If `function` is provided the expression is wrapped in a call to that function.\n        '
        with self._interpolation():
            if function:
                self.output.write(function)
                self.output.write('(')
            self._process_node(node.left, **kwargs)
            self.output.write(math_operator)
            self._process_node(node.right, **kwargs)
            if function:
                self.output.write(')')

    def _process_loop_helper(self, node, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Processes a loop helper e.g. {{ loop.first }} or {{ loop.index }}\n        '
        if node.attr == LOOP_HELPER_INDEX:
            self.output.write('(arguments[1] + 1)')
        elif node.attr == LOOP_HELPER_INDEX_0:
            self.output.write('arguments[1]')
        elif node.attr == LOOP_HELPER_FIRST:
            self.output.write('(arguments[1] == 0)')
        elif node.attr == LOOP_HELPER_LAST:
            self.output.write('(arguments[1] == arguments[2].length - 1)')
        elif node.attr == LOOP_HELPER_LENGTH:
            self.output.write('arguments[2].length')

    def _process_args(self, node, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        args = getattr(node, 'args', None)
        if not args:
            return
        for (i, item) in enumerate(args):
            self._process_node(item, **kwargs)
            if i < len(node.args) - 1:
                self.output.write(',')

    @contextlib.contextmanager
    def _execution(self):
        if False:
            i = 10
            return i + 15
        '\n        Context manager for executing some JavaScript inside a template.\n        '
        did_start_executing = False
        if self.state == STATE_DEFAULT:
            did_start_executing = True
            self.state = STATE_EXECUTING

        def close():
            if False:
                return 10
            if did_start_executing and self.state == STATE_EXECUTING:
                self.state = STATE_DEFAULT
        yield close
        close()

    @contextlib.contextmanager
    def _interpolation(self, safe=False):
        if False:
            return 10
        did_start_interpolating = False
        if self.state == STATE_DEFAULT:
            did_start_interpolating = True
            self.output.write('__result += "" + ')
            if safe is not True:
                self.output.write('__runtime.escape')
            self.output.write('((__tmp = (')
            self.state = STATE_INTERPOLATING

        def close():
            if False:
                return 10
            if did_start_interpolating and self.state == STATE_INTERPOLATING:
                self.output.write(')) == null ? "" : __tmp);')
                self.state = STATE_DEFAULT
        yield close
        close()

    @contextlib.contextmanager
    def _scoped_variables(self, nodes_list, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Context manager for creating scoped variables defined by the nodes in `nodes_list`.\n        These variables will be added to the context, and when the context manager exits the\n        context object will be restored to it's previous state.\n        "
        tmp_vars = []
        for node in nodes_list:
            is_assign_node = isinstance(node, nodes.Assign)
            name = node.target.name if is_assign_node else node.name
            tmp_var = next(self.temp_var_names)
            with self._execution():
                self.output.write('var %s = %s.%s;' % (tmp_var, self.context_name, name))
                self.output.write('%s.%s = ' % (self.context_name, name))
                if is_assign_node:
                    self._process_node(node.node, **kwargs)
                else:
                    self.output.write(node.name)
                self.output.write(';')
            tmp_vars.append((tmp_var, name))
        yield
        for (tmp_var, name) in tmp_vars:
            with self._execution():
                self.output.write('%s.%s = %s;' % (self.context_name, name, tmp_var))

    @contextlib.contextmanager
    def _python_bool_wrapper(self, **kwargs):
        if False:
            while True:
                i = 10
        use_python_bool_wrapper = kwargs.get('use_python_bool_wrapper')
        if use_python_bool_wrapper:
            self.output.write('__runtime.boolean(')
        with option(kwargs, use_python_bool_wrapper=False):
            yield kwargs
        if use_python_bool_wrapper:
            self.output.write(')')

def main(template_path, output=None, data_path=None):
    if False:
        while True:
            i = 10
    'Convert a jinja2 template to a JavaScript module.\n\n    template_path (Path): Path to .jijna file.\n    output (Optional[Path]): Path to output .js module (stdout if unset).\n    data_path (Optional[Path]): Optional JSON or YAML file with additional data\n        to be included in the JS module as the exported variable DATA.\n    '
    data = '{}'
    if data_path is not None:
        if data_path.suffix in ('.yml', '.yaml'):
            data = srsly.read_yaml(data_path)
        else:
            data = srsly.read_json(data_path)
        data = srsly.json_dumps(data)
    template_path = Path(template_path)
    tpl_file = template_path.parts[-1]
    compiler = JinjaToJS(template_path.parent, tpl_file, js_module_format='es6')
    header = f'// This file was auto-generated by {__file__} based on {tpl_file}'
    data_str = f'export const DATA = {data}'
    result = compiler.get_output()
    if output is not None:
        with output.open('w', encoding='utf8') as f:
            f.write(f'{header}\n{result}\n{data_str}')
        print(f'Updated {output.parts[-1]}')
    else:
        print(result)
if __name__ == '__main__':
    args = sys.argv[1:]
    if not len(args):
        raise ValueError('Need at least one argument: path to .jinja template')
    template_path = Path(args[0])
    output = Path(args[1]) if len(args) > 1 else None
    data_path = Path(args[2]) if len(args) > 2 else None
    main(template_path, output, data_path)