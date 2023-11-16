import distutils.ccompiler
import sys
import os
import subprocess
import glob
from distutils.sysconfig import customize_compiler
from gen_code import MappedObject
header = os.path.join('src', 'aubio.h')
output_path = os.path.join('python', 'gen')
source_header = '// this file is generated! do not modify\n#include "aubio-types.h"\n'
default_skip_objects = ['fft', 'pvoc', 'filter', 'filterbank', 'hist', 'parameter', 'scale', 'beattracking', 'resampler', 'peakpicker', 'pitchfcomb', 'pitchmcomb', 'pitchschmitt', 'pitchspecacf', 'pitchyin', 'pitchyinfft', 'pitchyinfast', 'sink', 'sink_apple_audio', 'sink_sndfile', 'sink_wavwrite', 'source', 'source_apple_audio', 'source_sndfile', 'source_avcodec', 'source_wavread', 'audio_unit', 'spectral_whitening', 'timestretch']

def get_preprocessor():
    if False:
        for i in range(10):
            print('nop')
    compiler_name = distutils.ccompiler.get_default_compiler()
    compiler = distutils.ccompiler.new_compiler(compiler=compiler_name)
    try:
        customize_compiler(compiler)
    except AttributeError as e:
        print('Warning: failed customizing compiler ({:s})'.format(repr(e)))
    if hasattr(compiler, 'initialize'):
        try:
            compiler.initialize()
        except ValueError as e:
            print('Warning: failed initializing compiler ({:s})'.format(repr(e)))
    cpp_cmd = None
    if hasattr(compiler, 'preprocessor'):
        cpp_cmd = compiler.preprocessor
    elif hasattr(compiler, 'compiler'):
        cpp_cmd = compiler.compiler.split()
        cpp_cmd += ['-E']
    elif hasattr(compiler, 'cc'):
        cpp_cmd = compiler.cc.split()
        cpp_cmd += ['-E']
    if len(cpp_cmd) > 1 and 'cl.exe' in cpp_cmd[-2]:
        plat = os.path.basename(os.path.dirname(cpp_cmd[-2]))
        if plat == 'x86_amd64':
            print('workaround on win64 to avoid empty pre-processor output')
            cpp_cmd[-2] = cpp_cmd[-2].replace('x86_amd64', '')
        elif True in ['amd64' in f for f in cpp_cmd]:
            print('warning: not using workaround for', cpp_cmd[0], plat)
    if not cpp_cmd:
        print("Warning: could not guess preprocessor, using env's CC")
        cpp_cmd = os.environ.get('CC', 'cc').split()
        cpp_cmd += ['-E']
    if 'emcc' in cpp_cmd:
        cpp_cmd += ['-x', 'c']
    return cpp_cmd

def get_c_declarations(header=header, usedouble=False):
    if False:
        while True:
            i = 10
    ' return a dense and preprocessed  string of all c declarations implied by aubio.h\n    '
    cpp_output = get_cpp_output(header=header, usedouble=usedouble)
    return filter_cpp_output(cpp_output)

def get_cpp_output(header=header, usedouble=False):
    if False:
        while True:
            i = 10
    ' find and run a C pre-processor on aubio.h '
    cpp_cmd = get_preprocessor()
    macros = [('AUBIO_UNSTABLE', 1)]
    if usedouble:
        macros += [('HAVE_AUBIO_DOUBLE', 1)]
    if not os.path.isfile(header):
        raise Exception('could not find include file ' + header)
    includes = [os.path.dirname(header)]
    cpp_cmd += distutils.ccompiler.gen_preprocess_options(macros, includes)
    cpp_cmd += [header]
    print('Running command: {:s}'.format(' '.join(cpp_cmd)))
    proc = subprocess.Popen(cpp_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    assert proc, 'Proc was none'
    cpp_output = proc.stdout.read()
    err_output = proc.stderr.read()
    if err_output:
        print('Warning: preprocessor produced errors or warnings:\n%s' % err_output)
    if not cpp_output:
        raise_msg = 'preprocessor output is empty! Running command ' + '"%s" failed' % ' '.join(cpp_cmd)
        if err_output:
            raise_msg += ' with stderr: "%s"' % err_output
        else:
            raise_msg += ' with no stdout or stderr'
        raise Exception(raise_msg)
    if not isinstance(cpp_output, list):
        cpp_output = [l.strip() for l in cpp_output.split('\n')]
    return cpp_output

def filter_cpp_output(cpp_raw_output):
    if False:
        return 10
    ' prepare cpp-output for parsing '
    cpp_output = filter(lambda y: len(y) > 1, cpp_raw_output)
    cpp_output = list(filter(lambda y: not y.startswith('#'), cpp_output))
    i = 1
    while 1:
        if i >= len(cpp_output):
            break
        if '{' in cpp_output[i - 1] and '}' not in cpp_output[i - 1] or ';' not in cpp_output[i - 1]:
            cpp_output[i] = cpp_output[i - 1] + ' ' + cpp_output[i]
            cpp_output.pop(i - 1)
        elif '}' in cpp_output[i]:
            cpp_output[i] = cpp_output[i - 1] + ' ' + cpp_output[i]
            cpp_output.pop(i - 1)
        else:
            i += 1
    tmp = []
    for l in cpp_output:
        tmp += [l.replace(' *', ' * ')]
    cpp_output = tmp
    return cpp_output

def get_cpp_objects_from_c_declarations(c_declarations, skip_objects=None):
    if False:
        for i in range(10):
            print('nop')
    if skip_objects is None:
        skip_objects = default_skip_objects
    typedefs = filter(lambda y: y.startswith('typedef struct _aubio'), c_declarations)
    cpp_objects = [a.split()[3][:-1] for a in typedefs]
    cpp_objects_filtered = filter(lambda y: not y[6:-2] in skip_objects, cpp_objects)
    return cpp_objects_filtered

def get_all_func_names_from_lib(lib):
    if False:
        for i in range(10):
            print('nop')
    ' return flat string of all function used in lib\n    '
    res = []
    for (_, v) in lib.items():
        if isinstance(v, dict):
            res += get_all_func_names_from_lib(v)
        elif isinstance(v, list):
            for elem in v:
                e = elem.split('(')
                if len(e) < 2:
                    continue
                fname_part = e[0].strip().split(' ')
                fname = fname_part[-1]
                if fname:
                    res += [fname]
                else:
                    raise NameError('gen_lib : weird function: ' + str(e))
    return res

def generate_lib_from_c_declarations(cpp_objects, c_declarations):
    if False:
        while True:
            i = 10
    ' returns a lib from given cpp_object names\n\n    a lib is a dict grouping functions by family (onset,pitch...)\n        each eement is itself a dict of functions grouped by puposes as : \n        struct, new, del, do, get, set and other\n    '
    lib = {}
    for o in cpp_objects:
        shortname = o
        if o[:6] == 'aubio_':
            shortname = o[6:-2]
        lib[shortname] = {'struct': [], 'new': [], 'del': [], 'do': [], 'rdo': [], 'get': [], 'set': [], 'other': []}
        lib[shortname]['longname'] = o
        lib[shortname]['shortname'] = shortname
        fullshortname = o[:-2]
        for fn in c_declarations:
            func_name = fn.split('(')[0].strip().split(' ')[-1]
            if func_name.startswith(fullshortname + '_') or func_name.endswith(fullshortname):
                if 'typedef struct ' in fn:
                    lib[shortname]['struct'].append(fn)
                elif '_do' in fn:
                    lib[shortname]['do'].append(fn)
                elif '_rdo' in fn:
                    lib[shortname]['rdo'].append(fn)
                elif 'new_' in fn:
                    lib[shortname]['new'].append(fn)
                elif 'del_' in fn:
                    lib[shortname]['del'].append(fn)
                elif '_get_' in fn:
                    lib[shortname]['get'].append(fn)
                elif '_set_' in fn:
                    lib[shortname]['set'].append(fn)
                else:
                    lib[shortname]['other'].append(fn)
    return lib

def print_c_declarations_results(lib, c_declarations):
    if False:
        while True:
            i = 10
    for fn in c_declarations:
        found = 0
        for o in lib:
            for family in lib[o]:
                if fn in lib[o][family]:
                    found = 1
        if found == 0:
            print('missing', fn)
    for o in lib:
        for family in lib[o]:
            if type(lib[o][family]) == str:
                print('{:15s} {:10s} {:s}'.format(o, family, lib[o][family]))
            elif len(lib[o][family]) == 1:
                print('{:15s} {:10s} {:s}'.format(o, family, lib[o][family][0]))
            else:
                print('{:15s} {:10s} {:s}'.format(o, family, lib[o][family]))

def generate_external(header=header, output_path=output_path, usedouble=False, overwrite=True):
    if False:
        print('Hello World!')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    elif not overwrite:
        return sorted(glob.glob(os.path.join(output_path, '*.c')))
    c_declarations = get_c_declarations(header, usedouble=usedouble)
    cpp_objects = get_cpp_objects_from_c_declarations(c_declarations)
    lib = generate_lib_from_c_declarations(cpp_objects, c_declarations)
    sources_list = []
    for o in lib:
        out = source_header
        mapped = MappedObject(lib[o], usedouble=usedouble)
        out += mapped.gen_code()
        output_file = os.path.join(output_path, 'gen-%s.c' % o)
        with open(output_file, 'w') as f:
            f.write(out)
            print('wrote %s' % output_file)
            sources_list.append(output_file)
    out = source_header
    out += '#include "aubio-generated.h"'
    check_types = '\n     ||  '.join(['PyType_Ready(&Py_%sType) < 0' % o for o in lib])
    out += '\n\nint generated_types_ready (void)\n{{\n  return ({pycheck_types});\n}}\n'.format(pycheck_types=check_types)
    add_types = ''.join(['\n  Py_INCREF (&Py_{name}Type);\n  PyModule_AddObject(m, "{name}", (PyObject *) & Py_{name}Type);'.format(name=o) for o in lib])
    out += '\n\nvoid add_generated_objects ( PyObject *m )\n{{\n{add_types}\n}}\n'.format(add_types=add_types)
    output_file = os.path.join(output_path, 'aubio-generated.c')
    with open(output_file, 'w') as f:
        f.write(out)
        print('wrote %s' % output_file)
        sources_list.append(output_file)
    objlist = ''.join(['extern PyTypeObject Py_%sType;\n' % p for p in lib])
    out = '// generated list of objects created with gen_external.py\n\n#include <Python.h>\n'
    if usedouble:
        out += '\n#ifndef HAVE_AUBIO_DOUBLE\n#define HAVE_AUBIO_DOUBLE 1\n#endif\n'
    out += '\n{objlist}\nint generated_objects ( void );\nvoid add_generated_objects( PyObject *m );\n'.format(objlist=objlist)
    output_file = os.path.join(output_path, 'aubio-generated.h')
    with open(output_file, 'w') as f:
        f.write(out)
        print('wrote %s' % output_file)
    return sorted(sources_list)
if __name__ == '__main__':
    if len(sys.argv) > 1:
        header = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    generate_external(header, output_path)