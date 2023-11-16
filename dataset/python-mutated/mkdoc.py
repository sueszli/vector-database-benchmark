import os
import sys
import platform
import re
import textwrap
from clang import cindex
from clang.cindex import CursorKind
from collections import OrderedDict
from threading import Thread, Semaphore
from multiprocessing import cpu_count
RECURSE_LIST = [CursorKind.TRANSLATION_UNIT, CursorKind.NAMESPACE, CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL, CursorKind.ENUM_DECL, CursorKind.CLASS_TEMPLATE]
PRINT_LIST = [CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL, CursorKind.ENUM_DECL, CursorKind.ENUM_CONSTANT_DECL, CursorKind.CLASS_TEMPLATE, CursorKind.FUNCTION_DECL, CursorKind.FUNCTION_TEMPLATE, CursorKind.CONVERSION_FUNCTION, CursorKind.CXX_METHOD, CursorKind.CONSTRUCTOR, CursorKind.FIELD_DECL]
CPP_OPERATORS = {'<=': 'le', '>=': 'ge', '==': 'eq', '!=': 'ne', '[]': 'array', '+=': 'iadd', '-=': 'isub', '*=': 'imul', '/=': 'idiv', '%=': 'imod', '&=': 'iand', '|=': 'ior', '^=': 'ixor', '<<=': 'ilshift', '>>=': 'irshift', '++': 'inc', '--': 'dec', '<<': 'lshift', '>>': 'rshift', '&&': 'land', '||': 'lor', '!': 'lnot', '~': 'bnot', '&': 'band', '|': 'bor', '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '%': 'mod', '<': 'lt', '>': 'gt', '=': 'assign', '()': 'call'}
CPP_OPERATORS = OrderedDict(sorted(CPP_OPERATORS.items(), key=lambda t: -len(t[0])))
job_count = cpu_count()
job_semaphore = Semaphore(job_count)
registered_names = dict()

def d(s):
    if False:
        return 10
    return s.decode('utf8')

def sanitize_name(name):
    if False:
        while True:
            i = 10
    global registered_names
    name = re.sub('type-parameter-0-([0-9]+)', 'T\\1', name)
    for (k, v) in CPP_OPERATORS.items():
        name = name.replace('operator%s' % k, 'operator_%s' % v)
    name = re.sub('<.*>', '', name)
    name = ''.join([ch if ch.isalnum() else '_' for ch in name])
    name = re.sub('_$', '', re.sub('_+', '_', name))
    if name in registered_names:
        registered_names[name] += 1
        name += '_' + str(registered_names[name])
    else:
        registered_names[name] = 1
    return '__doc_' + name

def process_comment(comment):
    if False:
        for i in range(10):
            print('nop')
    result = ''
    leading_spaces = float('inf')
    for s in comment.expandtabs(tabsize=4).splitlines():
        s = s.strip()
        if s.startswith('/*'):
            s = s[2:].lstrip('*')
        elif s.endswith('*/'):
            s = s[:-2].rstrip('*')
        elif s.startswith('///'):
            s = s[3:]
        if s.startswith('*'):
            s = s[1:]
        if len(s) > 0:
            leading_spaces = min(leading_spaces, len(s) - len(s.lstrip()))
        result += s + '\n'
    if leading_spaces != float('inf'):
        result2 = ''
        for s in result.splitlines():
            result2 += s[leading_spaces:] + '\n'
        result = result2
    cpp_group = '([\\w:]+)'
    param_group = '([\\[\\w:\\]]+)'
    s = result
    s = re.sub('\\\\c\\s+%s' % cpp_group, '``\\1``', s)
    s = re.sub('\\\\a\\s+%s' % cpp_group, '*\\1*', s)
    s = re.sub('\\\\e\\s+%s' % cpp_group, '*\\1*', s)
    s = re.sub('\\\\em\\s+%s' % cpp_group, '*\\1*', s)
    s = re.sub('\\\\b\\s+%s' % cpp_group, '**\\1**', s)
    s = re.sub('\\\\ingroup\\s+%s' % cpp_group, '', s)
    s = re.sub('\\\\param%s?\\s+%s' % (param_group, cpp_group), '\\n\\n$Parameter ``\\2``:\\n\\n', s)
    s = re.sub('\\\\tparam%s?\\s+%s' % (param_group, cpp_group), '\\n\\n$Template parameter ``\\2``:\\n\\n', s)
    for (in_, out_) in {'return': 'Returns', 'author': 'Author', 'authors': 'Authors', 'copyright': 'Copyright', 'date': 'Date', 'remark': 'Remark', 'sa': 'See also', 'see': 'See also', 'extends': 'Extends', 'throw': 'Throws', 'throws': 'Throws'}.items():
        s = re.sub('\\\\%s\\s*' % in_, '\\n\\n$%s:\\n\\n' % out_, s)
    s = re.sub('\\\\details\\s*', '\\n\\n', s)
    s = re.sub('\\\\brief\\s*', '', s)
    s = re.sub('\\\\short\\s*', '', s)
    s = re.sub('\\\\ref\\s*', '', s)
    s = re.sub('\\\\code\\s?(.*?)\\s?\\\\endcode', '```\\n\\1\\n```\\n', s, flags=re.DOTALL)
    s = re.sub('<tt>(.*?)</tt>', '``\\1``', s, flags=re.DOTALL)
    s = re.sub('<pre>(.*?)</pre>', '```\\n\\1\\n```\\n', s, flags=re.DOTALL)
    s = re.sub('<em>(.*?)</em>', '*\\1*', s, flags=re.DOTALL)
    s = re.sub('<b>(.*?)</b>', '**\\1**', s, flags=re.DOTALL)
    s = re.sub('\\\\f\\$(.*?)\\\\f\\$', '$\\1$', s, flags=re.DOTALL)
    s = re.sub('<li>', '\\n\\n* ', s)
    s = re.sub('</?ul>', '', s)
    s = re.sub('</li>', '\\n\\n', s)
    s = s.replace('``true``', '``True``')
    s = s.replace('``false``', '``False``')
    wrapper = textwrap.TextWrapper()
    wrapper.expand_tabs = True
    wrapper.replace_whitespace = True
    wrapper.drop_whitespace = True
    wrapper.width = 70
    wrapper.initial_indent = wrapper.subsequent_indent = ''
    result = ''
    in_code_segment = False
    for x in re.split('(```)', s):
        if x == '```':
            if not in_code_segment:
                result += '```\n'
            else:
                result += '\n```\n\n'
            in_code_segment = not in_code_segment
        elif in_code_segment:
            result += x.strip()
        else:
            for y in re.split('(?: *\\n *){2,}', x):
                wrapped = wrapper.fill(re.sub('\\s+', ' ', y).strip())
                if len(wrapped) > 0 and wrapped[0] == '$':
                    result += wrapped[1:] + '\n'
                    wrapper.initial_indent = wrapper.subsequent_indent = ' ' * 4
                else:
                    if len(wrapped) > 0:
                        result += wrapped + '\n\n'
                    wrapper.initial_indent = wrapper.subsequent_indent = ''
    return result.rstrip().lstrip('\n')

def extract(filename, node, prefix, output):
    if False:
        return 10
    num_extracted = 0
    if not (node.location.file is None or os.path.samefile(d(node.location.file.name), filename)):
        return 0
    if node.kind in RECURSE_LIST:
        sub_prefix = prefix
        if node.kind != CursorKind.TRANSLATION_UNIT:
            if len(sub_prefix) > 0:
                sub_prefix += '_'
            sub_prefix += d(node.spelling)
        for i in node.get_children():
            num_extracted += extract(filename, i, sub_prefix, output)
        if num_extracted == 0:
            return 0
    if node.kind in PRINT_LIST:
        comment = d(node.raw_comment) if node.raw_comment is not None else ''
        comment = process_comment(comment)
        sub_prefix = prefix
        if len(sub_prefix) > 0:
            sub_prefix += '_'
        if len(node.spelling) > 0:
            name = sanitize_name(sub_prefix + d(node.spelling))
            output.append('\nstatic const char *%s =%sR"doc(%s)doc";' % (name, '\n' if '\n' in comment else ' ', comment))
            num_extracted += 1
    return num_extracted

class ExtractionThread(Thread):

    def __init__(self, filename, parameters, output):
        if False:
            i = 10
            return i + 15
        Thread.__init__(self)
        self.filename = filename
        self.parameters = parameters
        self.output = output
        job_semaphore.acquire()

    def run(self):
        if False:
            while True:
                i = 10
        print('Processing "%s" ..' % self.filename, file=sys.stderr)
        try:
            index = cindex.Index(cindex.conf.lib.clang_createIndex(False, True))
            tu = index.parse(self.filename, self.parameters)
            extract(self.filename, tu.cursor, '', self.output)
        finally:
            job_semaphore.release()
if __name__ == '__main__':
    parameters = ['-x', 'c++', '-std=c++11']
    filenames = []
    if platform.system() == 'Darwin':
        dev_path = '/Applications/Xcode.app/Contents/Developer/'
        lib_dir = dev_path + 'Toolchains/XcodeDefault.xctoolchain/usr/lib/'
        sdk_dir = dev_path + 'Platforms/MacOSX.platform/Developer/SDKs'
        libclang = lib_dir + 'libclang.dylib'
        if os.path.exists(libclang):
            cindex.Config.set_library_path(os.path.dirname(libclang))
        if os.path.exists(sdk_dir):
            sysroot_dir = os.path.join(sdk_dir, next(os.walk(sdk_dir))[1][0])
            parameters.append('-isysroot')
            parameters.append(sysroot_dir)
    for item in sys.argv[1:]:
        if item.startswith('-'):
            parameters.append(item)
        else:
            filenames.append(item)
    if len(filenames) == 0:
        print('Syntax: %s [.. a list of header files ..]' % sys.argv[0])
        exit(-1)
    print('/*\n  This file contains docstrings for the Python bindings.\n  Do not edit! These were automatically extracted by mkdoc.py\n */\n\n#define __EXPAND(x)                                      x\n#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT\n#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))\n#define __CAT1(a, b)                                     a ## b\n#define __CAT2(a, b)                                     __CAT1(a, b)\n#define __DOC1(n1)                                       __doc_##n1\n#define __DOC2(n1, n2)                                   __doc_##n1##_##n2\n#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3\n#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4\n#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5\n#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6\n#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7\n#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))\n\n#if defined(__GNUG__)\n#pragma GCC diagnostic push\n#pragma GCC diagnostic ignored "-Wunused-variable"\n#endif\n')
    output = []
    for filename in filenames:
        thr = ExtractionThread(filename, parameters, output)
        thr.start()
    print('Waiting for jobs to finish ..', file=sys.stderr)
    for i in range(job_count):
        job_semaphore.acquire()
    output.sort()
    for l in output:
        print(l)
    print('\n#if defined(__GNUG__)\n#pragma GCC diagnostic pop\n#endif\n')