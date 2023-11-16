"""
Generate documentation for all registered implementation for lowering
using reStructured text.
"""
from subprocess import check_output
import os.path
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from collections import defaultdict
import inspect
from functools import partial
import numba
from numba.core.registry import cpu_target

def git_hash():
    if False:
        for i in range(10):
            print('nop')
    out = check_output(['git', 'log', "--pretty=format:'%H'", '-n', '1'])
    return out.decode('ascii').strip('\'"')

def get_func_name(fn):
    if False:
        i = 10
        return i + 15
    return getattr(fn, '__qualname__', fn.__name__)

def gather_function_info(backend):
    if False:
        return 10
    fninfos = defaultdict(list)
    basepath = os.path.dirname(os.path.dirname(numba.__file__))
    for (fn, osel) in backend._defns.items():
        for (sig, impl) in osel.versions:
            info = {}
            fninfos[fn].append(info)
            info['fn'] = fn
            info['sig'] = sig
            (code, firstlineno) = inspect.getsourcelines(impl)
            path = inspect.getsourcefile(impl)
            info['impl'] = {'name': get_func_name(impl), 'filename': os.path.relpath(path, start=basepath), 'lines': (firstlineno, firstlineno + len(code) - 1), 'docstring': impl.__doc__}
    return fninfos

def bind_file_to_print(fobj):
    if False:
        i = 10
        return i + 15
    return partial(print, file=fobj)

def format_signature(sig):
    if False:
        while True:
            i = 10

    def fmt(c):
        if False:
            i = 10
            return i + 15
        try:
            return c.__name__
        except AttributeError:
            return repr(c).strip('\'"')
    out = tuple(map(fmt, sig))
    return '`({0})`'.format(', '.join(out))
github_url = 'https://github.com/numba/numba/blob/{commit}/{path}#L{firstline}-L{lastline}'
description = '\nThis lists all lowering definition registered to the CPU target.\nEach subsection corresponds to a Python function that is supported by numba\nnopython mode. These functions have one or more lower implementation with\ndifferent signatures. The compiler chooses the most specific implementation\nfrom all overloads.\n'

def format_function_infos(fninfos):
    if False:
        while True:
            i = 10
    buf = StringIO()
    try:
        print = bind_file_to_print(buf)
        title_line = 'Lowering Listing'
        print(title_line)
        print('=' * len(title_line))
        print(description)
        commit = git_hash()

        def format_fname(fn):
            if False:
                while True:
                    i = 10
            try:
                fname = '{0}.{1}'.format(fn.__module__, get_func_name(fn))
            except AttributeError:
                fname = repr(fn)
            return (fn, fname)
        for (fn, fname) in sorted(map(format_fname, fninfos), key=lambda x: x[1]):
            impinfos = fninfos[fn]
            header_line = '``{0}``'.format(fname)
            print(header_line)
            print('-' * len(header_line))
            print()
            formatted_sigs = map(lambda x: format_signature(x['sig']), impinfos)
            sorted_impinfos = sorted(zip(formatted_sigs, impinfos), key=lambda x: x[0])
            col_signatures = ['Signature']
            col_urls = ['Definition']
            for (fmtsig, info) in sorted_impinfos:
                impl = info['impl']
                filename = impl['filename']
                lines = impl['lines']
                fname = impl['name']
                source = '{0} lines {1}-{2}'.format(filename, *lines)
                link = github_url.format(commit=commit, path=filename, firstline=lines[0], lastline=lines[1])
                url = '``{0}`` `{1} <{2}>`_'.format(fname, source, link)
                col_signatures.append(fmtsig)
                col_urls.append(url)
            max_width_col_sig = max(map(len, col_signatures))
            max_width_col_url = max(map(len, col_urls))
            padding = 2
            width_col_sig = padding * 2 + max_width_col_sig
            width_col_url = padding * 2 + max_width_col_url
            line_format = '{{0:^{0}}}  {{1:^{1}}}'.format(width_col_sig, width_col_url)
            print(line_format.format('=' * width_col_sig, '=' * width_col_url))
            print(line_format.format(col_signatures[0], col_urls[0]))
            print(line_format.format('=' * width_col_sig, '=' * width_col_url))
            for (sig, url) in zip(col_signatures[1:], col_urls[1:]):
                print(line_format.format(sig, url))
            print(line_format.format('=' * width_col_sig, '=' * width_col_url))
            print()
        return buf.getvalue()
    finally:
        buf.close()

def gen_lower_listing(path=None):
    if False:
        while True:
            i = 10
    '\n    Generate lowering listing to ``path`` or (if None) to stdout.\n    '
    cpu_backend = cpu_target.target_context
    cpu_backend.refresh()
    fninfos = gather_function_info(cpu_backend)
    out = format_function_infos(fninfos)
    if path is None:
        print(out)
    else:
        with open(path, 'w') as fobj:
            print(out, file=fobj)
if __name__ == '__main__':
    gen_lower_listing()