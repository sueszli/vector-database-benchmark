import subprocess
import pytest
from pyodide_build.pywasmcross import BuildArgs, calculate_exports, filter_objects, get_cmake_compiler_flags, get_library_output, handle_command_generate_args, replay_f2c, replay_genargs_handle_dashI

@pytest.fixture(scope='function')
def build_args():
    if False:
        for i in range(10):
            print('nop')
    yield BuildArgs(cflags='', cxxflags='', ldflags='', target_install_dir='', host_install_dir='', pythoninclude='python/include', exports='whole_archive')

def _args_wrapper(func):
    if False:
        while True:
            i = 10
    'Convert function to take as input / return a string instead of a\n    list of arguments\n\n    Also sets dryrun=True\n    '

    def _inner(line, *pargs):
        if False:
            i = 10
            return i + 15
        args = line.split()
        res = func(args, *pargs, dryrun=True)
        if hasattr(res, '__len__'):
            return ' '.join(res)
        else:
            return res
    return _inner
f2c_wrap = _args_wrapper(replay_f2c)

def generate_args(line: str, args: BuildArgs, is_link_cmd: bool=False) -> str:
    if False:
        while True:
            i = 10
    splitline = line.split()
    res = handle_command_generate_args(splitline, args, is_link_cmd)
    if res[0] in ('emcc', 'em++'):
        for arg in ['-Werror=implicit-function-declaration', '-Werror=mismatched-parameter-types', '-Werror=return-type']:
            assert arg in res
            res.remove(arg)
    if '-c' in splitline:
        if 'python/include' in res:
            include_index = res.index('python/include')
            del res[include_index]
            del res[include_index - 1]
    if is_link_cmd:
        arg = '-Wl,--fatal-warnings'
        assert arg in res
        res.remove(arg)
    return ' '.join(res)

def test_handle_command(build_args):
    if False:
        while True:
            i = 10
    args = build_args
    assert handle_command_generate_args(['gcc', '-print-multiarch'], args, True) == ['echo', 'wasm32-emscripten']
    proxied_commands = {'cc': 'emcc', 'c++': 'em++', 'gcc': 'emcc', 'ld': 'emcc', 'ar': 'emar', 'ranlib': 'emranlib', 'strip': 'emstrip', 'cmake': 'emcmake'}
    for (cmd, proxied_cmd) in proxied_commands.items():
        assert generate_args(cmd, args).split()[0] == proxied_cmd
    assert generate_args('gcc -c test.o -o test.so', args, True) == 'emcc -c test.o -o test.so'
    args = BuildArgs(cflags='-I./lib2', cxxflags='-std=c++11', ldflags='-lm', exports='whole_archive')
    assert generate_args('gcc -I./lib1 -c test.cpp -o test.o', args) == 'em++ -I./lib1 -c test.cpp -o test.o -I./lib2 -std=c++11'
    args = BuildArgs(cflags='', cxxflags='', ldflags='-lm', target_install_dir='', exports='whole_archive')
    assert generate_args('gcc -c test.o -o test.so', args, True) == 'emcc -c test.o -o test.so -lm'
    assert generate_args('gcc test.o -lbob -ljim -ljim -lbob -o test.so', args) == 'emcc test.o -lbob -ljim -o test.so'

def test_handle_command_ldflags(build_args):
    if False:
        return 10
    args = build_args
    assert generate_args('gcc -Wl,--strip-all,--as-needed -Wl,--sort-common,-z,now,-Bsymbolic-functions -c test.o -o test.so', args, True) == 'emcc -Wl,-z,now -c test.o -o test.so'

def test_replay_genargs_handle_dashI(monkeypatch):
    if False:
        while True:
            i = 10
    import sys
    mock_prefix = '/mock_prefix'
    mock_base_prefix = '/mock_base_prefix'
    monkeypatch.setattr(sys, 'prefix', mock_prefix)
    monkeypatch.setattr(sys, 'base_prefix', mock_base_prefix)
    target_dir = '/target'
    target_cpython_include = '/target/include/python3.11'
    assert replay_genargs_handle_dashI('-I/usr/include', target_dir) is None
    assert replay_genargs_handle_dashI(f'-I{mock_prefix}/include/python3.11', target_dir) == f'-I{target_cpython_include}'
    assert replay_genargs_handle_dashI(f'-I{mock_base_prefix}/include/python3.11', target_dir) == f'-I{target_cpython_include}'

def test_f2c():
    if False:
        return 10
    assert f2c_wrap('gfortran test.f') == 'gcc test.c'
    assert f2c_wrap('gcc test.c') is None
    assert f2c_wrap('gfortran --version') is None
    assert f2c_wrap('gfortran --shared -c test.o -o test.so') == 'gcc --shared -c test.o -o test.so'

def test_conda_unsupported_args(build_args):
    if False:
        i = 10
        return i + 15
    args = build_args
    assert generate_args('gcc -c test.o -B /compiler_compat -o test.so', args) == 'emcc -c test.o -o test.so'
    assert generate_args('gcc -c test.o -Wl,--sysroot=/ -o test.so', args) == 'emcc -c test.o -o test.so'

@pytest.mark.parametrize('line, expected', [([], []), (['obj1.o', 'obj2.o', 'slib1.so', 'slib2.so', 'lib1.a', 'lib2.a', '-o', 'test.so'], ['obj1.o', 'obj2.o', 'lib1.a', 'lib2.a']), (['@dir/link.txt', 'obj1.o', 'obj2.o', 'test.so'], ['@dir/link.txt', 'obj1.o', 'obj2.o'])])
def test_filter_objects(line, expected):
    if False:
        for i in range(10):
            print('nop')
    assert filter_objects(line) == expected

@pytest.mark.xfail(reason='FIXME: emcc is not available during test')
def test_exports_node(tmp_path):
    if False:
        return 10
    template = '\n        int l();\n\n        __attribute__((visibility("hidden")))\n        int f%s() {\n            return l();\n        }\n\n        __attribute__ ((visibility ("default")))\n        int g%s() {\n            return l();\n        }\n\n        int h%s(){\n            return l();\n        }\n        '
    (tmp_path / 'f1.c').write_text(template % (1, 1, 1))
    (tmp_path / 'f2.c').write_text(template % (2, 2, 2))
    subprocess.run(['emcc', '-c', tmp_path / 'f1.c', '-o', tmp_path / 'f1.o', '-fPIC'])
    subprocess.run(['emcc', '-c', tmp_path / 'f2.c', '-o', tmp_path / 'f2.o', '-fPIC'])
    assert set(calculate_exports([str(tmp_path / 'f1.o')], True)) == {'g1', 'h1'}
    assert set(calculate_exports([str(tmp_path / 'f1.o'), str(tmp_path / 'f2.o')], True)) == {'g1', 'h1', 'g2', 'h2'}
    subprocess.run(['emcc', '-c', tmp_path / 'f1.c', '-o', tmp_path / 'f1.o', '-fPIC', '-flto'])
    assert set(calculate_exports([str(tmp_path / 'f1.o')], True)) == {'f1', 'g1', 'h1'}

def test_get_cmake_compiler_flags():
    if False:
        while True:
            i = 10
    cmake_flags = ' '.join(get_cmake_compiler_flags())
    compiler_flags = ('CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER', 'CMAKE_C_COMPILER_AR', 'CMAKE_CXX_COMPILER_AR')
    for compiler_flag in compiler_flags:
        assert f'-D{compiler_flag}' in cmake_flags
    emscripten_compilers = ('emcc', 'em++', 'emar')
    for emscripten_compiler in emscripten_compilers:
        assert emscripten_compiler not in cmake_flags

def test_handle_command_cmake(build_args):
    if False:
        i = 10
        return i + 15
    args = build_args
    assert '--fresh' in handle_command_generate_args(['cmake', './'], args, False)
    build_cmd = ['cmake', '--build', '.--target', 'target']
    assert handle_command_generate_args(build_cmd, args, False) == build_cmd

def test_get_library_output():
    if False:
        i = 10
        return i + 15
    assert get_library_output(['test.so']) == 'test.so'
    assert get_library_output(['test.so.1.2.3']) == 'test.so.1.2.3'
    assert get_library_output(['test', 'test.a', 'test.o', 'test.c', 'test.cpp', 'test.h']) is None