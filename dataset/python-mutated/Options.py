from __future__ import absolute_import
import os
from .. import Utils

class ShouldBeFromDirective(object):
    known_directives = []

    def __init__(self, options_name, directive_name=None, disallow=False):
        if False:
            print('Hello World!')
        self.options_name = options_name
        self.directive_name = directive_name or options_name
        self.disallow = disallow
        self.known_directives.append(self)

    def __nonzero__(self):
        if False:
            print('Hello World!')
        self._bad_access()

    def __int__(self):
        if False:
            i = 10
            return i + 15
        self._bad_access()

    def _bad_access(self):
        if False:
            while True:
                i = 10
        raise RuntimeError(repr(self))

    def __repr__(self):
        if False:
            print('Hello World!')
        return "Illegal access of '%s' from Options module rather than directive '%s'" % (self.options_name, self.directive_name)
'\nThe members of this module are documented using autodata in\nCython/docs/src/reference/compilation.rst.\nSee https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#directive-autoattribute\nfor how autodata works.\nDescriptions of those members should start with a #:\nDonc forget to keep the docs in sync by removing and adding\nthe members in both this file and the .rst file.\n'
docstrings = True
embed_pos_in_docstring = False
pre_import = None
generate_cleanup_code = False
clear_to_none = True
annotate = False
annotate_coverage_xml = None
fast_fail = False
warning_errors = False
error_on_unknown_names = True
error_on_uninitialized = True
convert_range = True
cache_builtins = True
gcc_branch_hints = True
lookup_module_cpdef = False
embed = None
old_style_globals = ShouldBeFromDirective('old_style_globals')
cimport_from_pyx = False
buffer_max_dims = 8
closure_freelist_size = 8

def get_directive_defaults():
    if False:
        while True:
            i = 10
    for old_option in ShouldBeFromDirective.known_directives:
        value = globals().get(old_option.options_name)
        assert old_option.directive_name in _directive_defaults
        if not isinstance(value, ShouldBeFromDirective):
            if old_option.disallow:
                raise RuntimeError("Option '%s' must be set from directive '%s'" % (old_option.option_name, old_option.directive_name))
            else:
                _directive_defaults[old_option.directive_name] = value
    return _directive_defaults

def copy_inherited_directives(outer_directives, **new_directives):
    if False:
        print('Hello World!')
    new_directives_out = dict(outer_directives)
    for name in ('test_assert_path_exists', 'test_fail_if_path_exists', 'test_assert_c_code_has', 'test_fail_if_c_code_has'):
        new_directives_out.pop(name, None)
    new_directives_out.update(new_directives)
    return new_directives_out
_directive_defaults = {'binding': True, 'boundscheck': True, 'nonecheck': False, 'initializedcheck': True, 'embedsignature': False, 'embedsignature.format': 'c', 'auto_cpdef': False, 'auto_pickle': None, 'cdivision': False, 'cdivision_warnings': False, 'cpow': None, 'c_api_binop_methods': False, 'overflowcheck': False, 'overflowcheck.fold': True, 'always_allow_keywords': True, 'allow_none_for_extension_args': True, 'wraparound': True, 'ccomplex': False, 'callspec': '', 'nogil': False, 'gil': False, 'with_gil': False, 'profile': False, 'linetrace': False, 'emit_code_comments': True, 'annotation_typing': True, 'infer_types': None, 'infer_types.verbose': False, 'autotestdict': True, 'autotestdict.cdef': False, 'autotestdict.all': False, 'language_level': None, 'fast_getattr': False, 'py2_import': False, 'preliminary_late_includes_cy28': False, 'iterable_coroutine': False, 'c_string_type': 'bytes', 'c_string_encoding': '', 'type_version_tag': True, 'unraisable_tracebacks': True, 'old_style_globals': False, 'np_pythran': False, 'fast_gil': False, 'cpp_locals': False, 'legacy_implicit_noexcept': False, 'set_initial_path': None, 'warn': None, 'warn.undeclared': False, 'warn.unreachable': True, 'warn.maybe_uninitialized': False, 'warn.unused': False, 'warn.unused_arg': False, 'warn.unused_result': False, 'warn.multiple_declarators': True, 'show_performance_hints': True, 'optimize.inline_defnode_calls': True, 'optimize.unpack_method_calls': True, 'optimize.unpack_method_calls_in_pyinit': False, 'optimize.use_switch': True, 'remove_unreachable': True, 'control_flow.dot_output': '', 'control_flow.dot_annotate_defs': False, 'test_assert_path_exists': [], 'test_fail_if_path_exists': [], 'test_assert_c_code_has': [], 'test_fail_if_c_code_has': [], 'formal_grammar': False}
extra_warnings = {'warn.maybe_uninitialized': True, 'warn.unreachable': True, 'warn.unused': True}

def one_of(*args):
    if False:
        print('Hello World!')

    def validate(name, value):
        if False:
            for i in range(10):
                print('nop')
        if value not in args:
            raise ValueError("%s directive must be one of %s, got '%s'" % (name, args, value))
        else:
            return value
    return validate

def normalise_encoding_name(option_name, encoding):
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> normalise_encoding_name('c_string_encoding', 'ascii')\n    'ascii'\n    >>> normalise_encoding_name('c_string_encoding', 'AsCIi')\n    'ascii'\n    >>> normalise_encoding_name('c_string_encoding', 'us-ascii')\n    'ascii'\n    >>> normalise_encoding_name('c_string_encoding', 'utF8')\n    'utf8'\n    >>> normalise_encoding_name('c_string_encoding', 'utF-8')\n    'utf8'\n    >>> normalise_encoding_name('c_string_encoding', 'deFAuLT')\n    'default'\n    >>> normalise_encoding_name('c_string_encoding', 'default')\n    'default'\n    >>> normalise_encoding_name('c_string_encoding', 'SeriousLyNoSuch--Encoding')\n    'SeriousLyNoSuch--Encoding'\n    "
    if not encoding:
        return ''
    if encoding.lower() in ('default', 'ascii', 'utf8'):
        return encoding.lower()
    import codecs
    try:
        decoder = codecs.getdecoder(encoding)
    except LookupError:
        return encoding
    for name in ('ascii', 'utf8'):
        if codecs.getdecoder(name) == decoder:
            return name
    return encoding

class DEFER_ANALYSIS_OF_ARGUMENTS:
    pass
DEFER_ANALYSIS_OF_ARGUMENTS = DEFER_ANALYSIS_OF_ARGUMENTS()
directive_types = {'language_level': str, 'auto_pickle': bool, 'locals': dict, 'final': bool, 'collection_type': one_of('sequence'), 'nogil': DEFER_ANALYSIS_OF_ARGUMENTS, 'gil': DEFER_ANALYSIS_OF_ARGUMENTS, 'with_gil': None, 'internal': bool, 'infer_types': bool, 'binding': bool, 'cfunc': None, 'ccall': None, 'ufunc': None, 'cpow': bool, 'inline': None, 'staticmethod': None, 'cclass': None, 'no_gc_clear': bool, 'no_gc': bool, 'returns': type, 'exceptval': type, 'set_initial_path': str, 'freelist': int, 'c_string_type': one_of('bytes', 'bytearray', 'str', 'unicode'), 'c_string_encoding': normalise_encoding_name, 'trashcan': bool, 'total_ordering': None, 'dataclasses.dataclass': DEFER_ANALYSIS_OF_ARGUMENTS, 'dataclasses.field': DEFER_ANALYSIS_OF_ARGUMENTS, 'embedsignature.format': one_of('c', 'clinic', 'python')}
for (key, val) in _directive_defaults.items():
    if key not in directive_types:
        directive_types[key] = type(val)
directive_scopes = {'auto_pickle': ('module', 'cclass'), 'final': ('cclass', 'function'), 'ccomplex': ('module',), 'collection_type': ('cclass',), 'nogil': ('function', 'with statement'), 'gil': 'with statement', 'with_gil': ('function',), 'inline': ('function',), 'cfunc': ('function', 'with statement'), 'ccall': ('function', 'with statement'), 'returns': ('function',), 'exceptval': ('function',), 'locals': ('function',), 'staticmethod': ('function',), 'no_gc_clear': ('cclass',), 'no_gc': ('cclass',), 'internal': ('cclass',), 'cclass': ('class', 'cclass', 'with statement'), 'autotestdict': ('module',), 'autotestdict.all': ('module',), 'autotestdict.cdef': ('module',), 'set_initial_path': ('module',), 'test_assert_path_exists': ('function', 'class', 'cclass'), 'test_fail_if_path_exists': ('function', 'class', 'cclass'), 'test_assert_c_code_has': ('module',), 'test_fail_if_c_code_has': ('module',), 'freelist': ('cclass',), 'formal_grammar': ('module',), 'emit_code_comments': ('module',), 'c_string_type': ('module',), 'c_string_encoding': ('module',), 'type_version_tag': ('module', 'cclass'), 'language_level': ('module',), 'old_style_globals': ('module',), 'np_pythran': ('module',), 'preliminary_late_includes_cy28': ('module',), 'fast_gil': ('module',), 'iterable_coroutine': ('module', 'function'), 'trashcan': ('cclass',), 'total_ordering': ('class', 'cclass'), 'dataclasses.dataclass': ('class', 'cclass'), 'cpp_locals': ('module', 'function', 'cclass'), 'ufunc': ('function',), 'legacy_implicit_noexcept': ('module',), 'control_flow.dot_output': ('module',), 'control_flow.dot_annotate_defs': ('module',)}
immediate_decorator_directives = {'cfunc', 'ccall', 'cclass', 'dataclasses.dataclass', 'ufunc', 'inline', 'exceptval', 'returns', 'with_gil', 'freelist', 'no_gc', 'no_gc_clear', 'type_version_tag', 'final', 'auto_pickle', 'internal', 'collection_type', 'total_ordering', 'test_fail_if_path_exists', 'test_assert_path_exists'}

def parse_directive_value(name, value, relaxed_bool=False):
    if False:
        i = 10
        return i + 15
    "\n    Parses value as an option value for the given name and returns\n    the interpreted value. None is returned if the option does not exist.\n\n    >>> print(parse_directive_value('nonexisting', 'asdf asdfd'))\n    None\n    >>> parse_directive_value('boundscheck', 'True')\n    True\n    >>> parse_directive_value('boundscheck', 'true')\n    Traceback (most recent call last):\n       ...\n    ValueError: boundscheck directive must be set to True or False, got 'true'\n\n    >>> parse_directive_value('c_string_encoding', 'us-ascii')\n    'ascii'\n    >>> parse_directive_value('c_string_type', 'str')\n    'str'\n    >>> parse_directive_value('c_string_type', 'bytes')\n    'bytes'\n    >>> parse_directive_value('c_string_type', 'bytearray')\n    'bytearray'\n    >>> parse_directive_value('c_string_type', 'unicode')\n    'unicode'\n    >>> parse_directive_value('c_string_type', 'unnicode')\n    Traceback (most recent call last):\n    ValueError: c_string_type directive must be one of ('bytes', 'bytearray', 'str', 'unicode'), got 'unnicode'\n    "
    type = directive_types.get(name)
    if not type:
        return None
    orig_value = value
    if type is bool:
        value = str(value)
        if value == 'True':
            return True
        if value == 'False':
            return False
        if relaxed_bool:
            value = value.lower()
            if value in ('true', 'yes'):
                return True
            elif value in ('false', 'no'):
                return False
        raise ValueError("%s directive must be set to True or False, got '%s'" % (name, orig_value))
    elif type is int:
        try:
            return int(value)
        except ValueError:
            raise ValueError("%s directive must be set to an integer, got '%s'" % (name, orig_value))
    elif type is str:
        return str(value)
    elif callable(type):
        return type(name, value)
    else:
        assert False

def parse_directive_list(s, relaxed_bool=False, ignore_unknown=False, current_settings=None):
    if False:
        return 10
    '\n    Parses a comma-separated list of pragma options. Whitespace\n    is not considered.\n\n    >>> parse_directive_list(\'      \')\n    {}\n    >>> (parse_directive_list(\'boundscheck=True\') ==\n    ... {\'boundscheck\': True})\n    True\n    >>> parse_directive_list(\'  asdf\')\n    Traceback (most recent call last):\n       ...\n    ValueError: Expected "=" in option "asdf"\n    >>> parse_directive_list(\'boundscheck=hey\')\n    Traceback (most recent call last):\n       ...\n    ValueError: boundscheck directive must be set to True or False, got \'hey\'\n    >>> parse_directive_list(\'unknown=True\')\n    Traceback (most recent call last):\n       ...\n    ValueError: Unknown option: "unknown"\n    >>> warnings = parse_directive_list(\'warn.all=True\')\n    >>> len(warnings) > 1\n    True\n    >>> sum(warnings.values()) == len(warnings)  # all true.\n    True\n    '
    if current_settings is None:
        result = {}
    else:
        result = current_settings
    for item in s.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError('Expected "=" in option "%s"' % item)
        (name, value) = [s.strip() for s in item.strip().split('=', 1)]
        if name not in _directive_defaults:
            found = False
            if name.endswith('.all'):
                prefix = name[:-3]
                for directive in _directive_defaults:
                    if directive.startswith(prefix):
                        found = True
                        parsed_value = parse_directive_value(directive, value, relaxed_bool=relaxed_bool)
                        result[directive] = parsed_value
            if not found and (not ignore_unknown):
                raise ValueError('Unknown option: "%s"' % name)
        elif directive_types.get(name) is list:
            if name in result:
                result[name].append(value)
            else:
                result[name] = [value]
        else:
            parsed_value = parse_directive_value(name, value, relaxed_bool=relaxed_bool)
            result[name] = parsed_value
    return result

def parse_variable_value(value):
    if False:
        for i in range(10):
            print('nop')
    "\n    Parses value as an option value for the given name and returns\n    the interpreted value.\n\n    >>> parse_variable_value('True')\n    True\n    >>> parse_variable_value('true')\n    'true'\n    >>> parse_variable_value('us-ascii')\n    'us-ascii'\n    >>> parse_variable_value('str')\n    'str'\n    >>> parse_variable_value('123')\n    123\n    >>> parse_variable_value('1.23')\n    1.23\n\n    "
    if value == 'True':
        return True
    elif value == 'False':
        return False
    elif value == 'None':
        return None
    elif value.isdigit():
        return int(value)
    else:
        try:
            value = float(value)
        except Exception:
            pass
        return value

def parse_compile_time_env(s, current_settings=None):
    if False:
        while True:
            i = 10
    '\n    Parses a comma-separated list of pragma options. Whitespace\n    is not considered.\n\n    >>> parse_compile_time_env(\'      \')\n    {}\n    >>> (parse_compile_time_env(\'HAVE_OPENMP=True\') ==\n    ... {\'HAVE_OPENMP\': True})\n    True\n    >>> parse_compile_time_env(\'  asdf\')\n    Traceback (most recent call last):\n       ...\n    ValueError: Expected "=" in option "asdf"\n    >>> parse_compile_time_env(\'NUM_THREADS=4\') == {\'NUM_THREADS\': 4}\n    True\n    >>> parse_compile_time_env(\'unknown=anything\') == {\'unknown\': \'anything\'}\n    True\n    '
    if current_settings is None:
        result = {}
    else:
        result = current_settings
    for item in s.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError('Expected "=" in option "%s"' % item)
        (name, value) = [s.strip() for s in item.split('=', 1)]
        result[name] = parse_variable_value(value)
    return result

class CompilationOptions(object):
    """
    See default_options at the end of this module for a list of all possible
    options and CmdLine.usage and CmdLine.parse_command_line() for their
    meaning.
    """

    def __init__(self, defaults=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.include_path = []
        if defaults:
            if isinstance(defaults, CompilationOptions):
                defaults = defaults.__dict__
        else:
            defaults = default_options
        options = dict(defaults)
        options.update(kw)
        unknown_options = set(options) - set(default_options)
        unknown_options.difference_update(['include_path'])
        if unknown_options:
            message = 'got unknown compilation option%s, please remove: %s' % ('s' if len(unknown_options) > 1 else '', ', '.join(unknown_options))
            raise ValueError(message)
        directive_defaults = get_directive_defaults()
        directives = dict(options['compiler_directives'])
        unknown_directives = set(directives) - set(directive_defaults)
        if unknown_directives:
            message = 'got unknown compiler directive%s: %s' % ('s' if len(unknown_directives) > 1 else '', ', '.join(unknown_directives))
            raise ValueError(message)
        options['compiler_directives'] = directives
        if directives.get('np_pythran', False) and (not options['cplus']):
            import warnings
            warnings.warn('C++ mode forced when in Pythran mode!')
            options['cplus'] = True
        if 'language_level' not in kw and directives.get('language_level'):
            options['language_level'] = directives['language_level']
        elif not options.get('language_level'):
            options['language_level'] = directive_defaults.get('language_level')
        if 'formal_grammar' in directives and 'formal_grammar' not in kw:
            options['formal_grammar'] = directives['formal_grammar']
        if options['cache'] is True:
            options['cache'] = os.path.join(Utils.get_cython_cache_dir(), 'compiler')
        self.__dict__.update(options)

    def configure_language_defaults(self, source_extension):
        if False:
            return 10
        if source_extension == 'py':
            if self.compiler_directives.get('binding') is None:
                self.compiler_directives['binding'] = True

    def get_fingerprint(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a string that contains all the options that are relevant for cache invalidation.\n        '
        data = {}
        for (key, value) in self.__dict__.items():
            if key in ['show_version', 'errors_to_stderr', 'verbose', 'quiet']:
                continue
            elif key in ['output_file', 'output_dir']:
                continue
            elif key in ['depfile']:
                continue
            elif key in ['timestamps']:
                continue
            elif key in ['cache']:
                continue
            elif key in ['compiler_directives']:
                continue
            elif key in ['include_path']:
                continue
            elif key in ['working_path']:
                continue
            elif key in ['create_extension']:
                continue
            elif key in ['build_dir']:
                continue
            elif key in ['use_listing_file', 'generate_pxi', 'annotate', 'annotate_coverage_xml']:
                data[key] = value
            elif key in ['formal_grammar', 'evaluate_tree_assertions']:
                data[key] = value
            elif key in ['embedded_metadata', 'emit_linenums', 'c_line_in_traceback', 'gdb_debug', 'relative_path_in_code_position_comments']:
                data[key] = value
            elif key in ['cplus', 'language_level', 'compile_time_env', 'np_pythran']:
                data[key] = value
            elif key == ['capi_reexport_cincludes']:
                if self.capi_reexport_cincludes:
                    raise NotImplementedError('capi_reexport_cincludes is not compatible with Cython caching')
            elif key == ['common_utility_include_dir']:
                if self.common_utility_include_dir:
                    raise NotImplementedError('common_utility_include_dir is not compatible with Cython caching yet')
            else:
                data[key] = value

        def to_fingerprint(item):
            if False:
                return 10
            '\n            Recursively turn item into a string, turning dicts into lists with\n            deterministic ordering.\n            '
            if isinstance(item, dict):
                item = sorted([(repr(key), to_fingerprint(value)) for (key, value) in item.items()])
            return repr(item)
        return to_fingerprint(data)
default_options = dict(show_version=0, use_listing_file=0, errors_to_stderr=1, cplus=0, output_file=None, depfile=None, annotate=None, annotate_coverage_xml=None, generate_pxi=0, capi_reexport_cincludes=0, working_path='', timestamps=None, verbose=0, quiet=0, compiler_directives={}, embedded_metadata={}, evaluate_tree_assertions=False, emit_linenums=False, relative_path_in_code_position_comments=True, c_line_in_traceback=True, language_level=None, formal_grammar=False, gdb_debug=False, compile_time_env=None, module_name=None, common_utility_include_dir=None, output_dir=None, build_dir=None, cache=None, create_extension=None, np_pythran=False, legacy_implicit_noexcept=None)