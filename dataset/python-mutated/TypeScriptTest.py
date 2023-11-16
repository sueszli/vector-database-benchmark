import platform
import shutil
import subprocess
import sys
from pathlib import Path
tests_path = Path(__file__).parent.resolve()
root_path = tests_path.parent.parent.absolute()
is_windows = platform.system() == 'Windows'
flatc_exe = Path('flatc.exe' if is_windows else 'flatc')
if root_path in flatc_exe.parents:
    flatc_exe = flatc_exe.relative_to(root_path)
flatc_path = Path(root_path, flatc_exe)
assert flatc_path.exists(), 'Cannot find the flatc compiler ' + str(flatc_path)

def check_call(args, cwd=tests_path):
    if False:
        return 10
    subprocess.check_call(args, cwd=str(cwd), shell=is_windows)

def flatc(options, schema, prefix=None, include=None, data=None, cwd=tests_path):
    if False:
        print('Hello World!')
    print('Invoking flatc on schema ' + str(schema))
    cmd = [str(flatc_path)] + options
    if prefix:
        cmd += ['-o'] + [prefix]
    if include:
        cmd += ['-I'] + [include]
    cmd += [schema] if isinstance(schema, str) else schema
    if data:
        cmd += [data] if isinstance(data, str) else data
    check_call(cmd)

def esbuild(input, output):
    if False:
        i = 10
        return i + 15
    cmd = ['esbuild', input, '--outfile=' + output]
    cmd += ['--format=cjs', '--bundle', '--external:flatbuffers']
    check_call(cmd)
print('Removing node_modules/ directory...')
shutil.rmtree(Path(tests_path, 'node_modules'), ignore_errors=True)
check_call(['npm', 'install', '--silent'])
flatc(options=['--ts', '--reflect-names', '--gen-name-strings', '--gen-mutable', '--gen-object-api', '--ts-entry-points', '--ts-flat-files'], schema='../monster_test.fbs', include='../include_test')
esbuild('monster_test.ts', 'monster_test_generated.cjs')
flatc(options=['--gen-object-api', '-b'], schema='../monster_test.fbs', include='../include_test', data='../unicode_test.json')
flatc(options=['--ts', '--reflect-names', '--gen-name-strings', '--gen-mutable', '--gen-object-api', '--ts-entry-points', '--ts-flat-files'], schema='../union_vector/union_vector.fbs', prefix='union_vector')
esbuild('union_vector/union_vector.ts', 'union_vector/union_vector_generated.cjs')
flatc(options=['--ts', '--reflect-names', '--gen-name-strings'], schema='../optional_scalars.fbs')
flatc(options=['--ts', '--reflect-names', '--gen-name-strings', '--ts-no-import-ext'], schema='../optional_scalars.fbs', prefix='no_import_ext')
flatc(options=['--ts', '--reflect-names', '--gen-name-strings', '--gen-object-api', '--ts-entry-points', '--ts-flat-files'], schema='arrays_test_complex/arrays_test_complex.fbs', prefix='arrays_test_complex')
esbuild('arrays_test_complex/my-game/example.ts', 'arrays_test_complex/arrays_test_complex_generated.cjs')
flatc(options=['--ts', '--reflect-names', '--gen-name-strings', '--gen-mutable', '--gen-object-api', '--ts-entry-points', '--ts-flat-files'], schema=['typescript_keywords.fbs', 'test_dir/typescript_include.fbs', 'test_dir/typescript_transitive_include.fbs', '../../reflection/reflection.fbs'], include='../../')
esbuild('typescript_keywords.ts', 'typescript_keywords_generated.cjs')
flatc(options=['--ts', '--reflect-names', '--gen-name-strings', '--gen-mutable', '--gen-object-api', '--ts-entry-points', '--ts-flat-files'], schema='../union_underlying_type_test.fbs')
print('Running TypeScript Compiler...')
check_call(['tsc'])
print('Running TypeScript Compiler in old node resolution mode for no_import_ext...')
check_call(['tsc', '-p', './tsconfig.node.json'])
NODE_CMD = ['node']
print('Running TypeScript Tests...')
check_call(NODE_CMD + ['JavaScriptTest'])
check_call(NODE_CMD + ['JavaScriptUnionVectorTest'])
check_call(NODE_CMD + ['JavaScriptFlexBuffersTest'])
check_call(NODE_CMD + ['JavaScriptComplexArraysTest'])
check_call(NODE_CMD + ['JavaScriptUnionUnderlyingTypeTest'])
print('Running old v1 TypeScript Tests...')
check_call(NODE_CMD + ['JavaScriptTestv1.cjs', './monster_test_generated.cjs'])