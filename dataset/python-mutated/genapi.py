"""
Get API information encoded in C files.

See ``find_function`` for how functions should be formatted, and
``read_order`` for how the order of the functions should be
specified.

"""
import hashlib
import io
import os
import re
import sys
import importlib.util
import textwrap
from os.path import join

def get_processor():
    if False:
        while True:
            i = 10
    conv_template_path = os.path.join(os.path.dirname(__file__), '..', '..', 'distutils', 'conv_template.py')
    spec = importlib.util.spec_from_file_location('conv_template', conv_template_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.process_file
process_c_file = get_processor()
__docformat__ = 'restructuredtext'
API_FILES = [join('multiarray', 'alloc.c'), join('multiarray', 'abstractdtypes.c'), join('multiarray', 'arrayfunction_override.c'), join('multiarray', 'array_assign_array.c'), join('multiarray', 'array_assign_scalar.c'), join('multiarray', 'array_coercion.c'), join('multiarray', 'array_method.c'), join('multiarray', 'arrayobject.c'), join('multiarray', 'arraytypes.c.src'), join('multiarray', 'buffer.c'), join('multiarray', 'calculation.c'), join('multiarray', 'common_dtype.c'), join('multiarray', 'conversion_utils.c'), join('multiarray', 'convert.c'), join('multiarray', 'convert_datatype.c'), join('multiarray', 'ctors.c'), join('multiarray', 'datetime.c'), join('multiarray', 'datetime_busday.c'), join('multiarray', 'datetime_busdaycal.c'), join('multiarray', 'datetime_strings.c'), join('multiarray', 'descriptor.c'), join('multiarray', 'dlpack.c'), join('multiarray', 'dtypemeta.c'), join('multiarray', 'einsum.c.src'), join('multiarray', 'flagsobject.c'), join('multiarray', 'getset.c'), join('multiarray', 'item_selection.c'), join('multiarray', 'iterators.c'), join('multiarray', 'mapping.c'), join('multiarray', 'methods.c'), join('multiarray', 'multiarraymodule.c'), join('multiarray', 'nditer_api.c'), join('multiarray', 'nditer_constr.c'), join('multiarray', 'nditer_pywrap.c'), join('multiarray', 'nditer_templ.c.src'), join('multiarray', 'number.c'), join('multiarray', 'refcount.c'), join('multiarray', 'scalartypes.c.src'), join('multiarray', 'scalarapi.c'), join('multiarray', 'sequence.c'), join('multiarray', 'shape.c'), join('multiarray', 'strfuncs.c'), join('multiarray', 'usertypes.c'), join('umath', 'loops.c.src'), join('umath', 'ufunc_object.c'), join('umath', 'ufunc_type_resolution.c'), join('umath', 'reduction.c')]
THIS_DIR = os.path.dirname(__file__)
API_FILES = [os.path.join(THIS_DIR, '..', 'src', a) for a in API_FILES]

def file_in_this_dir(filename):
    if False:
        return 10
    return os.path.join(THIS_DIR, filename)

def remove_whitespace(s):
    if False:
        for i in range(10):
            print('nop')
    return ''.join(s.split())

def _repl(str):
    if False:
        for i in range(10):
            print('nop')
    return str.replace('Bool', 'npy_bool')

class MinVersion:

    def __init__(self, version):
        if False:
            i = 10
            return i + 15
        ' Version should be the normal NumPy version, e.g. "1.25" '
        (major, minor) = version.split('.')
        self.version = f'NPY_{major}_{minor}_API_VERSION'

    def __str__(self):
        if False:
            print('Hello World!')
        return self.version

    def add_guard(self, name, normal_define):
        if False:
            return 10
        'Wrap a definition behind a version guard'
        wrap = textwrap.dedent(f'\n            #if NPY_FEATURE_VERSION >= {self.version}\n            {{define}}\n            #endif')
        return wrap.format(define=normal_define)

class StealRef:

    def __init__(self, arg):
        if False:
            for i in range(10):
                print('nop')
        self.arg = arg

    def __str__(self):
        if False:
            return 10
        try:
            return ' '.join(('NPY_STEALS_REF_TO_ARG(%d)' % x for x in self.arg))
        except TypeError:
            return 'NPY_STEALS_REF_TO_ARG(%d)' % self.arg

class Function:

    def __init__(self, name, return_type, args, doc=''):
        if False:
            while True:
                i = 10
        self.name = name
        self.return_type = _repl(return_type)
        self.args = args
        self.doc = doc

    def _format_arg(self, typename, name):
        if False:
            for i in range(10):
                print('nop')
        if typename.endswith('*'):
            return typename + name
        else:
            return typename + ' ' + name

    def __str__(self):
        if False:
            while True:
                i = 10
        argstr = ', '.join([self._format_arg(*a) for a in self.args])
        if self.doc:
            doccomment = '/* %s */\n' % self.doc
        else:
            doccomment = ''
        return '%s%s %s(%s)' % (doccomment, self.return_type, self.name, argstr)

    def api_hash(self):
        if False:
            while True:
                i = 10
        m = hashlib.md5()
        m.update(remove_whitespace(self.return_type))
        m.update('\x00')
        m.update(self.name)
        m.update('\x00')
        for (typename, name) in self.args:
            m.update(remove_whitespace(typename))
            m.update('\x00')
        return m.hexdigest()[:8]

class ParseError(Exception):

    def __init__(self, filename, lineno, msg):
        if False:
            return 10
        self.filename = filename
        self.lineno = lineno
        self.msg = msg

    def __str__(self):
        if False:
            return 10
        return '%s:%s:%s' % (self.filename, self.lineno, self.msg)

def skip_brackets(s, lbrac, rbrac):
    if False:
        while True:
            i = 10
    count = 0
    for (i, c) in enumerate(s):
        if c == lbrac:
            count += 1
        elif c == rbrac:
            count -= 1
        if count == 0:
            return i
    raise ValueError("no match '%s' for '%s' (%r)" % (lbrac, rbrac, s))

def split_arguments(argstr):
    if False:
        i = 10
        return i + 15
    arguments = []
    current_argument = []
    i = 0

    def finish_arg():
        if False:
            print('Hello World!')
        if current_argument:
            argstr = ''.join(current_argument).strip()
            m = re.match('(.*(\\s+|\\*))(\\w+)$', argstr)
            if m:
                typename = m.group(1).strip()
                name = m.group(3)
            else:
                typename = argstr
                name = ''
            arguments.append((typename, name))
            del current_argument[:]
    while i < len(argstr):
        c = argstr[i]
        if c == ',':
            finish_arg()
        elif c == '(':
            p = skip_brackets(argstr[i:], '(', ')')
            current_argument += argstr[i:i + p]
            i += p - 1
        else:
            current_argument += c
        i += 1
    finish_arg()
    return arguments

def find_functions(filename, tag='API'):
    if False:
        print('Hello World!')
    "\n    Scan the file, looking for tagged functions.\n\n    Assuming ``tag=='API'``, a tagged function looks like::\n\n        /*API*/\n        static returntype*\n        function_name(argtype1 arg1, argtype2 arg2)\n        {\n        }\n\n    where the return type must be on a separate line, the function\n    name must start the line, and the opening ``{`` must start the line.\n\n    An optional documentation comment in ReST format may follow the tag,\n    as in::\n\n        /*API\n          This function does foo...\n         */\n    "
    if filename.endswith(('.c.src', '.h.src')):
        fo = io.StringIO(process_c_file(filename))
    else:
        fo = open(filename, 'r')
    functions = []
    return_type = None
    function_name = None
    function_args = []
    doclist = []
    (SCANNING, STATE_DOC, STATE_RETTYPE, STATE_NAME, STATE_ARGS) = list(range(5))
    state = SCANNING
    tagcomment = '/*' + tag
    for (lineno, line) in enumerate(fo):
        try:
            line = line.strip()
            if state == SCANNING:
                if line.startswith(tagcomment):
                    if line.endswith('*/'):
                        state = STATE_RETTYPE
                    else:
                        state = STATE_DOC
            elif state == STATE_DOC:
                if line.startswith('*/'):
                    state = STATE_RETTYPE
                else:
                    line = line.lstrip(' *')
                    doclist.append(line)
            elif state == STATE_RETTYPE:
                m = re.match('NPY_NO_EXPORT\\s+(.*)$', line)
                if m:
                    line = m.group(1)
                return_type = line
                state = STATE_NAME
            elif state == STATE_NAME:
                m = re.match('(\\w+)\\s*\\(', line)
                if m:
                    function_name = m.group(1)
                else:
                    raise ParseError(filename, lineno + 1, 'could not find function name')
                function_args.append(line[m.end():])
                state = STATE_ARGS
            elif state == STATE_ARGS:
                if line.startswith('{'):
                    fargs_str = ' '.join(function_args).rstrip()[:-1].rstrip()
                    fargs = split_arguments(fargs_str)
                    f = Function(function_name, return_type, fargs, '\n'.join(doclist))
                    functions.append(f)
                    return_type = None
                    function_name = None
                    function_args = []
                    doclist = []
                    state = SCANNING
                else:
                    function_args.append(line)
        except ParseError:
            raise
        except Exception as e:
            msg = 'see chained exception for details'
            raise ParseError(filename, lineno + 1, msg) from e
    fo.close()
    return functions

def write_file(filename, data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Write data to filename\n    Only write changed data to avoid updating timestamps unnecessarily\n    '
    if os.path.exists(filename):
        with open(filename) as f:
            if data == f.read():
                return
    with open(filename, 'w') as fid:
        fid.write(data)

class TypeApi:

    def __init__(self, name, index, ptr_cast, api_name, internal_type=None):
        if False:
            while True:
                i = 10
        self.index = index
        self.name = name
        self.ptr_cast = ptr_cast
        self.api_name = api_name
        self.internal_type = internal_type

    def define_from_array_api_string(self):
        if False:
            return 10
        return '#define %s (*(%s *)%s[%d])' % (self.name, self.ptr_cast, self.api_name, self.index)

    def array_api_define(self):
        if False:
            print('Hello World!')
        return '        (void *) &%s' % self.name

    def internal_define(self):
        if False:
            i = 10
            return i + 15
        if self.internal_type is None:
            return f'extern NPY_NO_EXPORT {self.ptr_cast} {self.name};\n'
        mangled_name = f'{self.name}Full'
        astr = f'extern NPY_NO_EXPORT {self.internal_type} {mangled_name};\n#define {self.name} (*({self.ptr_cast} *)(&{mangled_name}))\n'
        return astr

class GlobalVarApi:

    def __init__(self, name, index, type, api_name):
        if False:
            print('Hello World!')
        self.name = name
        self.index = index
        self.type = type
        self.api_name = api_name

    def define_from_array_api_string(self):
        if False:
            while True:
                i = 10
        return '#define %s (*(%s *)%s[%d])' % (self.name, self.type, self.api_name, self.index)

    def array_api_define(self):
        if False:
            print('Hello World!')
        return '        (%s *) &%s' % (self.type, self.name)

    def internal_define(self):
        if False:
            i = 10
            return i + 15
        astr = 'extern NPY_NO_EXPORT %(type)s %(name)s;\n' % {'type': self.type, 'name': self.name}
        return astr

class BoolValuesApi:

    def __init__(self, name, index, api_name):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.index = index
        self.type = 'PyBoolScalarObject'
        self.api_name = api_name

    def define_from_array_api_string(self):
        if False:
            return 10
        return '#define %s ((%s *)%s[%d])' % (self.name, self.type, self.api_name, self.index)

    def array_api_define(self):
        if False:
            print('Hello World!')
        return '        (void *) &%s' % self.name

    def internal_define(self):
        if False:
            print('Hello World!')
        astr = 'extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];\n'
        return astr

class FunctionApi:

    def __init__(self, name, index, annotations, return_type, args, api_name):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.index = index
        self.min_version = None
        self.annotations = []
        for annotation in annotations:
            if type(annotation).__name__ == 'StealRef':
                self.annotations.append(annotation)
            elif type(annotation).__name__ == 'MinVersion':
                if self.min_version is not None:
                    raise ValueError('Two minimum versions specified!')
                self.min_version = annotation
            else:
                raise ValueError(f'unknown annotation {annotation}')
        self.return_type = return_type
        self.args = args
        self.api_name = api_name

    def _argtypes_string(self):
        if False:
            print('Hello World!')
        if not self.args:
            return 'void'
        argstr = ', '.join([_repl(a[0]) for a in self.args])
        return argstr

    def define_from_array_api_string(self):
        if False:
            print('Hello World!')
        arguments = self._argtypes_string()
        define = textwrap.dedent(f'            #define {self.name} \\\n                    (*({self.return_type} (*)({arguments})) \\\n                {self.api_name}[{self.index}])')
        if self.min_version is not None:
            define = self.min_version.add_guard(self.name, define)
        return define

    def array_api_define(self):
        if False:
            i = 10
            return i + 15
        return '        (void *) %s' % self.name

    def internal_define(self):
        if False:
            return 10
        annstr = [str(a) for a in self.annotations]
        annstr = ' '.join(annstr)
        astr = 'NPY_NO_EXPORT %s %s %s \\\n       (%s);' % (annstr, self.return_type, self.name, self._argtypes_string())
        return astr

def order_dict(d):
    if False:
        print('Hello World!')
    'Order dict by its values.'
    o = list(d.items())

    def _key(x):
        if False:
            return 10
        return x[1] + (x[0],)
    return sorted(o, key=_key)

def merge_api_dicts(dicts):
    if False:
        return 10
    ret = {}
    for d in dicts:
        for (k, v) in d.items():
            ret[k] = v
    return ret

def check_api_dict(d):
    if False:
        print('Hello World!')
    'Check that an api dict is valid (does not use the same index twice)\n    and removed `__unused_indices__` from it (which is important only here)\n    '
    removed = set(d.pop('__unused_indices__', []))
    index_d = {k: v[0] for (k, v) in d.items()}
    revert_dict = {v: k for (k, v) in index_d.items()}
    if not len(revert_dict) == len(index_d):
        doubled = {}
        for (name, index) in index_d.items():
            try:
                doubled[index].append(name)
            except KeyError:
                doubled[index] = [name]
        fmt = 'Same index has been used twice in api definition: {}'
        val = ''.join(('\n\tindex {} -> {}'.format(index, names) for (index, names) in doubled.items() if len(names) != 1))
        raise ValueError(fmt.format(val))
    indexes = set(index_d.values())
    expected = set(range(len(indexes) + len(removed)))
    if not indexes.isdisjoint(removed):
        raise ValueError(f'API index used but marked unused: {indexes.intersection(removed)}')
    if indexes.union(removed) != expected:
        diff = expected.symmetric_difference(indexes.union(removed))
        msg = 'There are some holes in the API indexing: (symmetric diff is %s)' % diff
        raise ValueError(msg)

def get_api_functions(tagname, api_dict):
    if False:
        while True:
            i = 10
    'Parse source files to get functions tagged by the given tag.'
    functions = []
    for f in API_FILES:
        functions.extend(find_functions(f, tagname))
    dfunctions = [(api_dict[func.name][0], func) for func in functions]
    dfunctions.sort()
    return [a[1] for a in dfunctions]

def fullapi_hash(api_dicts):
    if False:
        return 10
    'Given a list of api dicts defining the numpy C API, compute a checksum\n    of the list of items in the API (as a string).'
    a = []
    for d in api_dicts:
        d = d.copy()
        d.pop('__unused_indices__', None)
        for (name, data) in order_dict(d):
            a.extend(name)
            a.extend(','.join(map(str, data)))
    return hashlib.md5(''.join(a).encode('ascii')).hexdigest()
VERRE = re.compile('(^0x[\\da-f]{8})\\s*=\\s*([\\da-f]{32})')

def get_versions_hash():
    if False:
        print('Hello World!')
    d = []
    file = os.path.join(os.path.dirname(__file__), 'cversions.txt')
    with open(file) as fid:
        for line in fid:
            m = VERRE.match(line)
            if m:
                d.append((int(m.group(1), 16), m.group(2)))
    return dict(d)

def main():
    if False:
        while True:
            i = 10
    tagname = sys.argv[1]
    order_file = sys.argv[2]
    functions = get_api_functions(tagname, order_file)
    m = hashlib.md5(tagname)
    for func in functions:
        print(func)
        ah = func.api_hash()
        m.update(ah)
        print(hex(int(ah, 16)))
    print(hex(int(m.hexdigest()[:8], 16)))
if __name__ == '__main__':
    main()