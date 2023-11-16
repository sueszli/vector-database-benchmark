"""Upgrader for Python scripts according to an API change specification."""
import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
FIND_OPEN = re.compile('^\\s*(\\[).*$')
FIND_STRING_CHARS = re.compile('[\'\\"]')
INFO = 'INFO'
WARNING = 'WARNING'
ERROR = 'ERROR'
ImportRename = collections.namedtuple('ImportRename', ['new_name', 'excluded_prefixes'])

def full_name_node(name, ctx=ast.Load()):
    if False:
        return 10
    'Make an Attribute or Name node for name.\n\n  Translate a qualified name into nested Attribute nodes (and a Name node).\n\n  Args:\n    name: The name to translate to a node.\n    ctx: What context this name is used in. Defaults to Load()\n\n  Returns:\n    A Name or Attribute node.\n  '
    names = name.split('.')
    names.reverse()
    node = ast.Name(id=names.pop(), ctx=ast.Load())
    while names:
        node = ast.Attribute(value=node, attr=names.pop(), ctx=ast.Load())
    node.ctx = ctx
    return node

def get_arg_value(node, arg_name, arg_pos=None):
    if False:
        print('Hello World!')
    "Get the value of an argument from a ast.Call node.\n\n  This function goes through the positional and keyword arguments to check\n  whether a given argument was used, and if so, returns its value (the node\n  representing its value).\n\n  This cannot introspect *args or **args, but it safely handles *args in\n  Python3.5+.\n\n  Args:\n    node: The ast.Call node to extract arg values from.\n    arg_name: The name of the argument to extract.\n    arg_pos: The position of the argument (in case it's passed as a positional\n      argument).\n\n  Returns:\n    A tuple (arg_present, arg_value) containing a boolean indicating whether\n    the argument is present, and its value in case it is.\n  "
    if arg_name is not None:
        for kw in node.keywords:
            if kw.arg == arg_name:
                return (True, kw.value)
    if arg_pos is not None:
        idx = 0
        for arg in node.args:
            if sys.version_info[:2] >= (3, 5) and isinstance(arg, ast.Starred):
                continue
            if idx == arg_pos:
                return (True, arg)
            idx += 1
    return (False, None)

def uses_star_args_in_call(node):
    if False:
        for i in range(10):
            print('nop')
    'Check if an ast.Call node uses arbitrary-length positional *args.\n\n  This function works with the AST call node format of Python3.5+\n  as well as the different AST format of earlier versions of Python.\n\n  Args:\n    node: The ast.Call node to check arg values for.\n\n  Returns:\n    True if the node uses starred variadic positional args or keyword args.\n    False if it does not.\n  '
    if sys.version_info[:2] >= (3, 5):
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                return True
    elif node.starargs:
        return True
    return False

def uses_star_kwargs_in_call(node):
    if False:
        for i in range(10):
            print('nop')
    'Check if an ast.Call node uses arbitrary-length **kwargs.\n\n  This function works with the AST call node format of Python3.5+\n  as well as the different AST format of earlier versions of Python.\n\n  Args:\n    node: The ast.Call node to check arg values for.\n\n  Returns:\n    True if the node uses starred variadic positional args or keyword args.\n    False if it does not.\n  '
    if sys.version_info[:2] >= (3, 5):
        for keyword in node.keywords:
            if keyword.arg is None:
                return True
    elif node.kwargs:
        return True
    return False

def uses_star_args_or_kwargs_in_call(node):
    if False:
        print('Hello World!')
    'Check if an ast.Call node uses arbitrary-length *args or **kwargs.\n\n  This function works with the AST call node format of Python3.5+\n  as well as the different AST format of earlier versions of Python.\n\n  Args:\n    node: The ast.Call node to check arg values for.\n\n  Returns:\n    True if the node uses starred variadic positional args or keyword args.\n    False if it does not.\n  '
    return uses_star_args_in_call(node) or uses_star_kwargs_in_call(node)

def excluded_from_module_rename(module, import_rename_spec):
    if False:
        for i in range(10):
            print('nop')
    'Check if this module import should not be renamed.\n\n  Args:\n    module: (string) module name.\n    import_rename_spec: ImportRename instance.\n\n  Returns:\n    True if this import should not be renamed according to the\n    import_rename_spec.\n  '
    for excluded_prefix in import_rename_spec.excluded_prefixes:
        if module.startswith(excluded_prefix):
            return True
    return False

class APIChangeSpec:
    """This class defines the transformations that need to happen.

  This class must provide the following fields:

  * `function_keyword_renames`: maps function names to a map of old -> new
    argument names
  * `symbol_renames`: maps function names to new function names
  * `change_to_function`: a set of function names that have changed (for
    notifications)
  * `function_reorders`: maps functions whose argument order has changed to the
    list of arguments in the new order
  * `function_warnings`: maps full names of functions to warnings that will be
    printed out if the function is used. (e.g. tf.nn.convolution())
  * `function_transformers`: maps function names to custom handlers
  * `module_deprecations`: maps module names to warnings that will be printed
    if the module is still used after all other transformations have run
  * `import_renames`: maps import name (must be a short name without '.')
    to ImportRename instance.

  For an example, see `TFAPIChangeSpec`.
  """

    def preprocess(self, root_node):
        if False:
            return 10
        'Preprocess a parse tree. Return a preprocessed node, logs and errors.'
        return (root_node, [], [])

    def clear_preprocessing(self):
        if False:
            return 10
        'Restore this APIChangeSpec to before it preprocessed a file.\n\n    This is needed if preprocessing a file changed any rewriting rules.\n    '
        pass

class NoUpdateSpec(APIChangeSpec):
    """A specification of an API change which doesn't change anything."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.function_handle = {}
        self.function_reorders = {}
        self.function_keyword_renames = {}
        self.symbol_renames = {}
        self.function_warnings = {}
        self.change_to_function = {}
        self.module_deprecations = {}
        self.function_transformers = {}
        self.import_renames = {}

class _PastaEditVisitor(ast.NodeVisitor):
    """AST Visitor that processes function calls.

  Updates function calls from old API version to new API version using a given
  change spec.
  """

    def __init__(self, api_change_spec):
        if False:
            i = 10
            return i + 15
        self._api_change_spec = api_change_spec
        self._log = []
        self._stack = []

    def visit(self, node):
        if False:
            return 10
        self._stack.append(node)
        super(_PastaEditVisitor, self).visit(node)
        self._stack.pop()

    @property
    def errors(self):
        if False:
            while True:
                i = 10
        return [log for log in self._log if log[0] == ERROR]

    @property
    def warnings(self):
        if False:
            return 10
        return [log for log in self._log if log[0] == WARNING]

    @property
    def warnings_and_errors(self):
        if False:
            i = 10
            return i + 15
        return [log for log in self._log if log[0] in (WARNING, ERROR)]

    @property
    def info(self):
        if False:
            i = 10
            return i + 15
        return [log for log in self._log if log[0] == INFO]

    @property
    def log(self):
        if False:
            i = 10
            return i + 15
        return self._log

    def add_log(self, severity, lineno, col, msg):
        if False:
            while True:
                i = 10
        self._log.append((severity, lineno, col, msg))
        print('%s line %d:%d: %s' % (severity, lineno, col, msg))

    def add_logs(self, logs):
        if False:
            for i in range(10):
                print('nop')
        'Record a log and print it.\n\n    The log should be a tuple `(severity, lineno, col_offset, msg)`, which will\n    be printed and recorded. It is part of the log available in the `self.log`\n    property.\n\n    Args:\n      logs: The logs to add. Must be a list of tuples\n        `(severity, lineno, col_offset, msg)`.\n    '
        self._log.extend(logs)
        for log in logs:
            print('%s line %d:%d: %s' % log)

    def _get_applicable_entries(self, transformer_field, full_name, name):
        if False:
            i = 10
            return i + 15
        'Get all list entries indexed by name that apply to full_name or name.'
        function_transformers = getattr(self._api_change_spec, transformer_field, {})
        glob_name = '*.' + name if name else None
        transformers = []
        if full_name in function_transformers:
            transformers.append(function_transformers[full_name])
        if glob_name in function_transformers:
            transformers.append(function_transformers[glob_name])
        if '*' in function_transformers:
            transformers.append(function_transformers['*'])
        return transformers

    def _get_applicable_dict(self, transformer_field, full_name, name):
        if False:
            return 10
        'Get all dict entries indexed by name that apply to full_name or name.'
        function_transformers = getattr(self._api_change_spec, transformer_field, {})
        glob_name = '*.' + name if name else None
        transformers = function_transformers.get('*', {}).copy()
        transformers.update(function_transformers.get(glob_name, {}))
        transformers.update(function_transformers.get(full_name, {}))
        return transformers

    def _get_full_name(self, node):
        if False:
            i = 10
            return i + 15
        'Traverse an Attribute node to generate a full name, e.g., "tf.foo.bar".\n\n    This is the inverse of `full_name_node`.\n\n    Args:\n      node: A Node of type Attribute.\n\n    Returns:\n      a \'.\'-delimited full-name or None if node was not Attribute or Name.\n      i.e. `foo()+b).bar` returns None, while `a.b.c` would return "a.b.c".\n    '
        curr = node
        items = []
        while not isinstance(curr, ast.Name):
            if not isinstance(curr, ast.Attribute):
                return None
            items.append(curr.attr)
            curr = curr.value
        items.append(curr.id)
        return '.'.join(reversed(items))

    def _maybe_add_warning(self, node, full_name):
        if False:
            print('Hello World!')
        'Adds an error to be printed about full_name at node.'
        function_warnings = self._api_change_spec.function_warnings
        if full_name in function_warnings:
            (level, message) = function_warnings[full_name]
            message = message.replace('<function name>', full_name)
            self.add_log(level, node.lineno, node.col_offset, '%s requires manual check. %s' % (full_name, message))
            return True
        else:
            return False

    def _maybe_add_module_deprecation_warning(self, node, full_name, whole_name):
        if False:
            i = 10
            return i + 15
        'Adds a warning if full_name is a deprecated module.'
        warnings = self._api_change_spec.module_deprecations
        if full_name in warnings:
            (level, message) = warnings[full_name]
            message = message.replace('<function name>', whole_name)
            self.add_log(level, node.lineno, node.col_offset, 'Using member %s in deprecated module %s. %s' % (whole_name, full_name, message))
            return True
        else:
            return False

    def _maybe_add_call_warning(self, node, full_name, name):
        if False:
            print('Hello World!')
        'Print a warning when specific functions are called with selected args.\n\n    The function _print_warning_for_function matches the full name of the called\n    function, e.g., tf.foo.bar(). This function matches the function name that\n    is called, as long as the function is an attribute. For example,\n    `tf.foo.bar()` and `foo.bar()` are matched, but not `bar()`.\n\n    Args:\n      node: ast.Call object\n      full_name: The precomputed full name of the callable, if one exists, None\n        otherwise.\n      name: The precomputed name of the callable, if one exists, None otherwise.\n\n    Returns:\n      Whether an error was recorded.\n    '
        warned = False
        if isinstance(node.func, ast.Attribute):
            warned = self._maybe_add_warning(node, '*.' + name)
        arg_warnings = self._get_applicable_dict('function_arg_warnings', full_name, name)
        variadic_args = uses_star_args_or_kwargs_in_call(node)
        for ((kwarg, arg), (level, warning)) in sorted(arg_warnings.items()):
            (present, _) = get_arg_value(node, kwarg, arg) or variadic_args
            if present:
                warned = True
                warning_message = warning.replace('<function name>', full_name or name)
                template = '%s called with %s argument, requires manual check: %s'
                if variadic_args:
                    template = '%s called with *args or **kwargs that may include %s, requires manual check: %s'
                self.add_log(level, node.lineno, node.col_offset, template % (full_name or name, kwarg, warning_message))
        return warned

    def _maybe_rename(self, parent, node, full_name):
        if False:
            print('Hello World!')
        'Replace node (Attribute or Name) with a node representing full_name.'
        new_name = self._api_change_spec.symbol_renames.get(full_name, None)
        if new_name:
            self.add_log(INFO, node.lineno, node.col_offset, 'Renamed %r to %r' % (full_name, new_name))
            new_node = full_name_node(new_name, node.ctx)
            ast.copy_location(new_node, node)
            pasta.ast_utils.replace_child(parent, node, new_node)
            return True
        else:
            return False

    def _maybe_change_to_function_call(self, parent, node, full_name):
        if False:
            print('Hello World!')
        'Wraps node (typically, an Attribute or Expr) in a Call.'
        if full_name in self._api_change_spec.change_to_function:
            if not isinstance(parent, ast.Call):
                new_node = ast.Call(node, [], [])
                pasta.ast_utils.replace_child(parent, node, new_node)
                ast.copy_location(new_node, node)
                self.add_log(INFO, node.lineno, node.col_offset, 'Changed %r to a function call' % full_name)
                return True
        return False

    def _maybe_add_arg_names(self, node, full_name):
        if False:
            for i in range(10):
                print('nop')
        'Make args into keyword args if function called full_name requires it.'
        function_reorders = self._api_change_spec.function_reorders
        if full_name in function_reorders:
            if uses_star_args_in_call(node):
                self.add_log(WARNING, node.lineno, node.col_offset, '(Manual check required) upgrading %s may require re-ordering the call arguments, but it was passed variable-length positional *args. The upgrade script cannot handle these automatically.' % full_name)
            reordered = function_reorders[full_name]
            new_args = []
            new_keywords = []
            idx = 0
            for arg in node.args:
                if sys.version_info[:2] >= (3, 5) and isinstance(arg, ast.Starred):
                    continue
                keyword_arg = reordered[idx]
                if keyword_arg:
                    new_keywords.append(ast.keyword(arg=keyword_arg, value=arg))
                else:
                    new_args.append(arg)
                idx += 1
            if new_keywords:
                self.add_log(INFO, node.lineno, node.col_offset, 'Added keywords to args of function %r' % full_name)
                node.args = new_args
                node.keywords = new_keywords + (node.keywords or [])
                return True
        return False

    def _maybe_modify_args(self, node, full_name, name):
        if False:
            return 10
        'Rename keyword args if the function called full_name requires it.'
        renamed_keywords = self._get_applicable_dict('function_keyword_renames', full_name, name)
        if not renamed_keywords:
            return False
        if uses_star_kwargs_in_call(node):
            self.add_log(WARNING, node.lineno, node.col_offset, '(Manual check required) upgrading %s may require renaming or removing call arguments, but it was passed variable-length *args or **kwargs. The upgrade script cannot handle these automatically.' % (full_name or name))
        modified = False
        new_keywords = []
        for keyword in node.keywords:
            argkey = keyword.arg
            if argkey in renamed_keywords:
                modified = True
                if renamed_keywords[argkey] is None:
                    lineno = getattr(keyword, 'lineno', node.lineno)
                    col_offset = getattr(keyword, 'col_offset', node.col_offset)
                    self.add_log(INFO, lineno, col_offset, 'Removed argument %s for function %s' % (argkey, full_name or name))
                else:
                    keyword.arg = renamed_keywords[argkey]
                    lineno = getattr(keyword, 'lineno', node.lineno)
                    col_offset = getattr(keyword, 'col_offset', node.col_offset)
                    self.add_log(INFO, lineno, col_offset, 'Renamed keyword argument for %s from %s to %s' % (full_name, argkey, renamed_keywords[argkey]))
                    new_keywords.append(keyword)
            else:
                new_keywords.append(keyword)
        if modified:
            node.keywords = new_keywords
        return modified

    def visit_Call(self, node):
        if False:
            i = 10
            return i + 15
        'Handle visiting a call node in the AST.\n\n    Args:\n      node: Current Node\n    '
        assert self._stack[-1] is node
        full_name = self._get_full_name(node.func)
        if full_name:
            name = full_name.split('.')[-1]
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            name = None
        self._maybe_add_call_warning(node, full_name, name)
        self._maybe_add_arg_names(node, full_name)
        self._maybe_modify_args(node, full_name, name)
        transformers = self._get_applicable_entries('function_transformers', full_name, name)
        parent = self._stack[-2]
        if transformers:
            if uses_star_args_or_kwargs_in_call(node):
                self.add_log(WARNING, node.lineno, node.col_offset, '(Manual check required) upgrading %s may require modifying call arguments, but it was passed variable-length *args or **kwargs. The upgrade script cannot handle these automatically.' % (full_name or name))
        for transformer in transformers:
            logs = []
            new_node = transformer(parent, node, full_name, name, logs)
            self.add_logs(logs)
            if new_node and new_node is not node:
                pasta.ast_utils.replace_child(parent, node, new_node)
                node = new_node
                self._stack[-1] = node
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if False:
            return 10
        'Handle bare Attributes i.e. [tf.foo, tf.bar].'
        assert self._stack[-1] is node
        full_name = self._get_full_name(node)
        if full_name:
            parent = self._stack[-2]
            self._maybe_add_warning(node, full_name)
            if self._maybe_rename(parent, node, full_name):
                return
            if self._maybe_change_to_function_call(parent, node, full_name):
                return
            i = 2
            while isinstance(self._stack[-i], ast.Attribute):
                i += 1
            whole_name = pasta.dump(self._stack[-(i - 1)])
            self._maybe_add_module_deprecation_warning(node, full_name, whole_name)
        self.generic_visit(node)

    def visit_Import(self, node):
        if False:
            return 10
        'Handle visiting an import node in the AST.\n\n    Args:\n      node: Current Node\n    '
        new_aliases = []
        import_updated = False
        import_renames = getattr(self._api_change_spec, 'import_renames', {})
        max_submodule_depth = getattr(self._api_change_spec, 'max_submodule_depth', 1)
        inserts_after_imports = getattr(self._api_change_spec, 'inserts_after_imports', {})
        for import_alias in node.names:
            all_import_components = import_alias.name.split('.')
            found_update = False
            for i in reversed(list(range(1, max_submodule_depth + 1))):
                import_component = all_import_components[0]
                for j in range(1, min(i, len(all_import_components))):
                    import_component += '.' + all_import_components[j]
                import_rename_spec = import_renames.get(import_component, None)
                if not import_rename_spec or excluded_from_module_rename(import_alias.name, import_rename_spec):
                    continue
                new_name = import_rename_spec.new_name + import_alias.name[len(import_component):]
                new_asname = import_alias.asname
                if not new_asname and '.' not in import_alias.name:
                    new_asname = import_alias.name
                new_alias = ast.alias(name=new_name, asname=new_asname)
                new_aliases.append(new_alias)
                import_updated = True
                found_update = True
                full_import = (import_alias.name, import_alias.asname)
                insert_offset = 1
                for line_to_insert in inserts_after_imports.get(full_import, []):
                    assert self._stack[-1] is node
                    parent = self._stack[-2]
                    new_line_node = pasta.parse(line_to_insert)
                    ast.copy_location(new_line_node, node)
                    parent.body.insert(parent.body.index(node) + insert_offset, new_line_node)
                    insert_offset += 1
                    old_suffix = pasta.base.formatting.get(node, 'suffix')
                    if old_suffix is None:
                        old_suffix = os.linesep
                    if os.linesep not in old_suffix:
                        pasta.base.formatting.set(node, 'suffix', old_suffix + os.linesep)
                    pasta.base.formatting.set(new_line_node, 'prefix', pasta.base.formatting.get(node, 'prefix'))
                    pasta.base.formatting.set(new_line_node, 'suffix', os.linesep)
                    self.add_log(INFO, node.lineno, node.col_offset, 'Adding `%s` after import of %s' % (new_line_node, import_alias.name))
                if found_update:
                    break
            if not found_update:
                new_aliases.append(import_alias)
        if import_updated:
            assert self._stack[-1] is node
            parent = self._stack[-2]
            new_node = ast.Import(new_aliases)
            ast.copy_location(new_node, node)
            pasta.ast_utils.replace_child(parent, node, new_node)
            self.add_log(INFO, node.lineno, node.col_offset, 'Changed import from %r to %r.' % (pasta.dump(node), pasta.dump(new_node)))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if False:
            print('Hello World!')
        'Handle visiting an import-from node in the AST.\n\n    Args:\n      node: Current Node\n    '
        if not node.module:
            self.generic_visit(node)
            return
        from_import = node.module
        from_import_first_component = from_import.split('.')[0]
        import_renames = getattr(self._api_change_spec, 'import_renames', {})
        import_rename_spec = import_renames.get(from_import_first_component, None)
        if not import_rename_spec:
            self.generic_visit(node)
            return
        updated_aliases = []
        same_aliases = []
        for import_alias in node.names:
            full_module_name = '%s.%s' % (from_import, import_alias.name)
            if excluded_from_module_rename(full_module_name, import_rename_spec):
                same_aliases.append(import_alias)
            else:
                updated_aliases.append(import_alias)
        if not updated_aliases:
            self.generic_visit(node)
            return
        assert self._stack[-1] is node
        parent = self._stack[-2]
        new_from_import = import_rename_spec.new_name + from_import[len(from_import_first_component):]
        updated_node = ast.ImportFrom(new_from_import, updated_aliases, node.level)
        ast.copy_location(updated_node, node)
        pasta.ast_utils.replace_child(parent, node, updated_node)
        additional_import_log = ''
        if same_aliases:
            same_node = ast.ImportFrom(from_import, same_aliases, node.level, col_offset=node.col_offset, lineno=node.lineno)
            ast.copy_location(same_node, node)
            parent.body.insert(parent.body.index(updated_node), same_node)
            pasta.base.formatting.set(same_node, 'prefix', pasta.base.formatting.get(updated_node, 'prefix'))
            additional_import_log = ' and %r' % pasta.dump(same_node)
        self.add_log(INFO, node.lineno, node.col_offset, 'Changed import from %r to %r%s.' % (pasta.dump(node), pasta.dump(updated_node), additional_import_log))
        self.generic_visit(node)

class AnalysisResult:
    """This class represents an analysis result and how it should be logged.

  This class must provide the following fields:

  * `log_level`: The log level to which this detection should be logged
  * `log_message`: The message that should be logged for this detection

  For an example, see `VersionedTFImport`.
  """

class APIAnalysisSpec:
    """This class defines how `AnalysisResult`s should be generated.

  It specifies how to map imports and symbols to `AnalysisResult`s.

  This class must provide the following fields:

  * `symbols_to_detect`: maps function names to `AnalysisResult`s
  * `imports_to_detect`: maps imports represented as (full module name, alias)
    tuples to `AnalysisResult`s
    notifications)

  For an example, see `TFAPIImportAnalysisSpec`.
  """

class PastaAnalyzeVisitor(_PastaEditVisitor):
    """AST Visitor that looks for specific API usage without editing anything.

  This is used before any rewriting is done to detect if any symbols are used
  that require changing imports or disabling rewriting altogether.
  """

    def __init__(self, api_analysis_spec):
        if False:
            return 10
        super(PastaAnalyzeVisitor, self).__init__(NoUpdateSpec())
        self._api_analysis_spec = api_analysis_spec
        self._results = []

    @property
    def results(self):
        if False:
            while True:
                i = 10
        return self._results

    def add_result(self, analysis_result):
        if False:
            print('Hello World!')
        self._results.append(analysis_result)

    def visit_Attribute(self, node):
        if False:
            print('Hello World!')
        'Handle bare Attributes i.e. [tf.foo, tf.bar].'
        full_name = self._get_full_name(node)
        if full_name:
            detection = self._api_analysis_spec.symbols_to_detect.get(full_name, None)
            if detection:
                self.add_result(detection)
                self.add_log(detection.log_level, node.lineno, node.col_offset, detection.log_message)
        self.generic_visit(node)

    def visit_Import(self, node):
        if False:
            return 10
        'Handle visiting an import node in the AST.\n\n    Args:\n      node: Current Node\n    '
        for import_alias in node.names:
            full_import = (import_alias.name, import_alias.asname)
            detection = self._api_analysis_spec.imports_to_detect.get(full_import, None)
            if detection:
                self.add_result(detection)
                self.add_log(detection.log_level, node.lineno, node.col_offset, detection.log_message)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if False:
            i = 10
            return i + 15
        'Handle visiting an import-from node in the AST.\n\n    Args:\n      node: Current Node\n    '
        if not node.module:
            self.generic_visit(node)
            return
        from_import = node.module
        for import_alias in node.names:
            full_module_name = '%s.%s' % (from_import, import_alias.name)
            full_import = (full_module_name, import_alias.asname)
            detection = self._api_analysis_spec.imports_to_detect.get(full_import, None)
            if detection:
                self.add_result(detection)
                self.add_log(detection.log_level, node.lineno, node.col_offset, detection.log_message)
        self.generic_visit(node)

class ASTCodeUpgrader:
    """Handles upgrading a set of Python files using a given API change spec."""

    def __init__(self, api_change_spec):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(api_change_spec, APIChangeSpec):
            raise TypeError('Must pass APIChangeSpec to ASTCodeUpgrader, got %s' % type(api_change_spec))
        self._api_change_spec = api_change_spec

    def process_file(self, in_filename, out_filename, no_change_to_outfile_on_error=False):
        if False:
            for i in range(10):
                print('nop')
        'Process the given python file for incompatible changes.\n\n    Args:\n      in_filename: filename to parse\n      out_filename: output file to write to\n      no_change_to_outfile_on_error: not modify the output file on errors\n    Returns:\n      A tuple representing number of files processed, log of actions, errors\n    '
        with open(in_filename, 'r') as in_file, tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
            ret = self.process_opened_file(in_filename, in_file, out_filename, temp_file)
        if no_change_to_outfile_on_error and ret[0] == 0:
            os.remove(temp_file.name)
        else:
            shutil.move(temp_file.name, out_filename)
        return ret

    def format_log(self, log, in_filename):
        if False:
            while True:
                i = 10
        log_string = '%d:%d: %s: %s' % (log[1], log[2], log[0], log[3])
        if in_filename:
            return in_filename + ':' + log_string
        else:
            return log_string

    def update_string_pasta(self, text, in_filename):
        if False:
            return 10
        'Updates a file using pasta.'
        try:
            t = pasta.parse(text)
        except (SyntaxError, ValueError, TypeError):
            log = ['ERROR: Failed to parse.\n' + traceback.format_exc()]
            return (0, '', log, [])
        (t, preprocess_logs, preprocess_errors) = self._api_change_spec.preprocess(t)
        visitor = _PastaEditVisitor(self._api_change_spec)
        visitor.visit(t)
        self._api_change_spec.clear_preprocessing()
        logs = [self.format_log(log, None) for log in preprocess_logs + visitor.log]
        errors = [self.format_log(error, in_filename) for error in preprocess_errors + visitor.warnings_and_errors]
        return (1, pasta.dump(t), logs, errors)

    def _format_log(self, log, in_filename, out_filename):
        if False:
            print('Hello World!')
        text = '-' * 80 + '\n'
        text += 'Processing file %r\n outputting to %r\n' % (in_filename, out_filename)
        text += '-' * 80 + '\n\n'
        text += '\n'.join(log) + '\n'
        text += '-' * 80 + '\n\n'
        return text

    def process_opened_file(self, in_filename, in_file, out_filename, out_file):
        if False:
            i = 10
            return i + 15
        'Process the given python file for incompatible changes.\n\n    This function is split out to facilitate StringIO testing from\n    tf_upgrade_test.py.\n\n    Args:\n      in_filename: filename to parse\n      in_file: opened file (or StringIO)\n      out_filename: output file to write to\n      out_file: opened file (or StringIO)\n    Returns:\n      A tuple representing number of files processed, log of actions, errors\n    '
        lines = in_file.readlines()
        (processed_file, new_file_content, log, process_errors) = self.update_string_pasta(''.join(lines), in_filename)
        if out_file and processed_file:
            out_file.write(new_file_content)
        return (processed_file, self._format_log(log, in_filename, out_filename), process_errors)

    def process_tree(self, root_directory, output_root_directory, copy_other_files):
        if False:
            return 10
        'Processes upgrades on an entire tree of python files in place.\n\n    Note that only Python files. If you have custom code in other languages,\n    you will need to manually upgrade those.\n\n    Args:\n      root_directory: Directory to walk and process.\n      output_root_directory: Directory to use as base.\n      copy_other_files: Copy files that are not touched by this converter.\n\n    Returns:\n      A tuple of files processed, the report string for all files, and a dict\n        mapping filenames to errors encountered in that file.\n    '
        if output_root_directory == root_directory:
            return self.process_tree_inplace(root_directory)
        if output_root_directory and os.path.exists(output_root_directory):
            print('Output directory %r must not already exist.' % output_root_directory)
            sys.exit(1)
        norm_root = os.path.split(os.path.normpath(root_directory))
        norm_output = os.path.split(os.path.normpath(output_root_directory))
        if norm_root == norm_output:
            print('Output directory %r same as input directory %r' % (root_directory, output_root_directory))
            sys.exit(1)
        files_to_process = []
        files_to_copy = []
        for (dir_name, _, file_list) in os.walk(root_directory):
            py_files = [f for f in file_list if f.endswith('.py')]
            copy_files = [f for f in file_list if not f.endswith('.py')]
            for filename in py_files:
                fullpath = os.path.join(dir_name, filename)
                fullpath_output = os.path.join(output_root_directory, os.path.relpath(fullpath, root_directory))
                files_to_process.append((fullpath, fullpath_output))
            if copy_other_files:
                for filename in copy_files:
                    fullpath = os.path.join(dir_name, filename)
                    fullpath_output = os.path.join(output_root_directory, os.path.relpath(fullpath, root_directory))
                    files_to_copy.append((fullpath, fullpath_output))
        file_count = 0
        tree_errors = {}
        report = ''
        report += '=' * 80 + '\n'
        report += 'Input tree: %r\n' % root_directory
        report += '=' * 80 + '\n'
        for (input_path, output_path) in files_to_process:
            output_directory = os.path.dirname(output_path)
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)
            if os.path.islink(input_path):
                link_target = os.readlink(input_path)
                link_target_output = os.path.join(output_root_directory, os.path.relpath(link_target, root_directory))
                if (link_target, link_target_output) in files_to_process:
                    os.symlink(link_target_output, output_path)
                else:
                    report += 'Copying symlink %s without modifying its target %s' % (input_path, link_target)
                    os.symlink(link_target, output_path)
                continue
            file_count += 1
            (_, l_report, l_errors) = self.process_file(input_path, output_path)
            tree_errors[input_path] = l_errors
            report += l_report
        for (input_path, output_path) in files_to_copy:
            output_directory = os.path.dirname(output_path)
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)
            shutil.copy(input_path, output_path)
        return (file_count, report, tree_errors)

    def process_tree_inplace(self, root_directory):
        if False:
            print('Hello World!')
        'Process a directory of python files in place.'
        files_to_process = []
        for (dir_name, _, file_list) in os.walk(root_directory):
            py_files = [os.path.join(dir_name, f) for f in file_list if f.endswith('.py')]
            files_to_process += py_files
        file_count = 0
        tree_errors = {}
        report = ''
        report += '=' * 80 + '\n'
        report += 'Input tree: %r\n' % root_directory
        report += '=' * 80 + '\n'
        for path in files_to_process:
            if os.path.islink(path):
                report += 'Skipping symlink %s.\n' % path
                continue
            file_count += 1
            (_, l_report, l_errors) = self.process_file(path, path)
            tree_errors[path] = l_errors
            report += l_report
        return (file_count, report, tree_errors)