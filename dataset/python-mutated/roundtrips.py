"""Run end-to-end cross-language roundtrip tests for all code examples."""
from __future__ import annotations
import argparse
import multiprocessing
import os
import subprocess
import sys
import time
from os import listdir
from os.path import isfile, join
opt_out_run = {'any_values': ['cpp', 'rust'], 'custom_data': ['cpp'], 'extra_values': ['cpp', 'rust'], 'image_advanced': ['cpp', 'rust'], 'log_line': ['cpp', 'rust', 'py'], 'quick_start_spawn': ['cpp'], 'timelines_example': ['py', 'cpp', 'rust'], 'roundtrips': ['cpp', 'py', 'rust']}
opt_out_compare = {'arrow3d_simple': ['cpp', 'py', 'rust'], 'asset3d_out_of_tree': ['cpp', 'py', 'rust'], 'mesh3d_partial_updates': ['cpp', 'py', 'rust'], 'pinhole_simple': ['cpp', 'py', 'rust'], 'point2d_random': ['cpp', 'py', 'rust'], 'point3d_random': ['cpp', 'py', 'rust'], 'quick_start_connect': ['cpp', 'py', 'rust'], 'quick_start_spawn': ['cpp', 'py', 'rust'], 'scalar_multiple_plots': ['cpp'], 'tensor_simple': ['cpp', 'py', 'rust']}
extra_args = {'asset3d_simple': [f'{os.path.dirname(__file__)}/../assets/cube.glb'], 'asset3d_out_of_tree': [f'{os.path.dirname(__file__)}/../assets/cube.glb']}

def run(args: list[str], *, env: dict[str, str] | None=None, timeout: int | None=None, cwd: str | None=None) -> None:
    if False:
        print('Hello World!')
    print(f'> {subprocess.list2cmdline(args)}')
    result = subprocess.run(args, env=env, cwd=cwd, timeout=timeout, check=False, capture_output=True, text=True)
    assert result.returncode == 0, f'{subprocess.list2cmdline(args)} failed with exit-code {result.returncode}. Output:\n{result.stdout}\n{result.stderr}'

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Run end-to-end cross-language roundtrip tests for all API examples')
    parser.add_argument('--no-py', action='store_true', help='Skip Python tests')
    parser.add_argument('--no-cpp', action='store_true', help='Skip C++ tests')
    parser.add_argument('--no-py-build', action='store_true', help='Skip building rerun-sdk for Python')
    parser.add_argument('--no-cpp-build', action='store_true', help='Skip cmake configure and ahead of time build for rerun_c & rerun_cpp')
    parser.add_argument('--full-dump', action='store_true', help='Dump both rrd files as tables')
    parser.add_argument('--release', action='store_true', help='Run cargo invocations with --release and CMake with `-DCMAKE_BUILD_TYPE=Release` & `--config Release`')
    parser.add_argument('--target', type=str, default=None, help='Target used for cargo invocations')
    parser.add_argument('--target-dir', type=str, default=None, help='Target directory used for cargo invocations')
    parser.add_argument('example', nargs='*', type=str, default=None, help='Run only the specified examples')
    args = parser.parse_args()
    build_env = os.environ.copy()
    if 'RUST_LOG' in build_env:
        del build_env['RUST_LOG']
    if args.no_py:
        pass
    elif args.no_py_build:
        print('Skipping building python rerun-sdk - assuming it is already built and up-to-date!')
    else:
        print('----------------------------------------------------------')
        print('Building rerun-sdk for Python…')
        start_time = time.time()
        run(['just', 'py-build', '--quiet'], env=build_env)
        elapsed = time.time() - start_time
        print(f'rerun-sdk for Python built in {elapsed:.1f} seconds')
        print('')
    if args.no_cpp:
        pass
    elif args.no_cpp_build:
        print('Skipping cmake configure & build for rerun_c & rerun_cpp - assuming it is already built and up-to-date!')
    else:
        print('----------------------------------------------------------')
        print('Build rerun_c & rerun_cpp…')
        start_time = time.time()
        os.makedirs('build', exist_ok=True)
        build_type = 'Debug'
        if args.release:
            build_type = 'Release'
        configure_args = ['cmake', f'-DCMAKE_BUILD_TYPE={build_type}', '-DCMAKE_COMPILE_WARNING_AS_ERROR=ON', '..']
        run(configure_args, env=build_env, cwd='build')
        cmake_build('rerun_sdk', args.release)
        elapsed = time.time() - start_time
        print(f'rerun-sdk for C++ built in {elapsed:.1f} seconds')
        print('')
    if len(args.example) > 0:
        examples = args.example
    else:
        dir = os.path.dirname(__file__)
        files = [f for f in listdir(dir) if isfile(join(dir, f))]
        examples = [filename for (filename, extension) in [os.path.splitext(file) for file in files] if extension == '.cpp' and (not args.no_cpp) or (extension == '.py' and (not args.no_py)) or extension == '.rs']
    examples = list(set(examples))
    examples.sort()
    print('----------------------------------------------------------')
    print(f'Running {len(examples)} examples…')
    active_languages = ['rust']
    if not args.no_cpp:
        active_languages.append('cpp')
    if not args.no_py:
        active_languages.append('py')
    if not args.no_cpp_build:
        for example in examples:
            example_opt_out_entirely = opt_out_run.get(example, [])
            if 'cpp' in example_opt_out_entirely:
                continue
            run_example(example, 'cpp', args)
    with multiprocessing.Pool() as pool:
        jobs = []
        for example in examples:
            example_opt_out_entirely = opt_out_run.get(example, [])
            for language in active_languages:
                if language in example_opt_out_entirely or language == 'cpp':
                    continue
                job = pool.apply_async(run_example, (example, language, args))
                jobs.append(job)
        print(f'Waiting for {len(jobs)} runs to finish…')
        for job in jobs:
            job.get()
    print('----------------------------------------------------------')
    print(f'Comparing {len(examples)} examples…')
    for example in examples:
        print()
        print('----------------------------------------------------------')
        print(f"Comparing example '{example}'…")
        example_opt_out_entirely = opt_out_run.get(example, [])
        example_opt_out_compare = opt_out_compare.get(example, [])
        if 'rust' in example_opt_out_entirely:
            continue
        cpp_output_path = f'docs/code-examples/{example}_cpp.rrd'
        python_output_path = f'docs/code-examples/{example}_py.rrd'
        rust_output_path = f'docs/code-examples/{example}_rust.rrd'
        if 'cpp' in active_languages and 'cpp' not in example_opt_out_entirely and ('cpp' not in example_opt_out_compare):
            run_comparison(cpp_output_path, rust_output_path, args.full_dump)
        if 'py' in active_languages and 'py' not in example_opt_out_entirely and ('py' not in example_opt_out_compare):
            run_comparison(python_output_path, rust_output_path, args.full_dump)
    print()
    print('----------------------------------------------------------')
    print('All tests passed!')

def run_example(example: str, language: str, args: argparse.Namespace) -> None:
    if False:
        return 10
    if language == 'cpp':
        cpp_output_path = run_roundtrip_cpp(example, args.release)
        check_non_empty_rrd(cpp_output_path)
    elif language == 'py':
        python_output_path = run_roundtrip_python(example)
        check_non_empty_rrd(python_output_path)
    elif language == 'rust':
        rust_output_path = run_roundtrip_rust(example, args.release, args.target, args.target_dir)
        check_non_empty_rrd(rust_output_path)
    else:
        assert False, f'Unknown language: {language}'

def roundtrip_env(*, save_path: str | None=None) -> dict[str, str]:
    if False:
        i = 10
        return i + 15
    env = os.environ.copy()
    env['RERUN_FLUSH_NUM_ROWS'] = '0'
    env['RERUN_STRICT'] = '1'
    env['RERUN_PANIC_ON_WARN'] = '1'
    if save_path:
        env['_RERUN_TEST_FORCE_SAVE'] = save_path
    return env

def run_roundtrip_python(example: str) -> str:
    if False:
        return 10
    main_path = f'docs/code-examples/{example}.py'
    output_path = f'docs/code-examples/{example}_py.rrd'
    python_executable = sys.executable
    if python_executable is None:
        python_executable = 'python3'
    cmd = [python_executable, main_path] + (extra_args.get(example) or [])
    env = roundtrip_env(save_path=output_path)
    run(cmd, env=env, timeout=30)
    return output_path

def run_roundtrip_rust(example: str, release: bool, target: str | None, target_dir: str | None) -> str:
    if False:
        for i in range(10):
            print('nop')
    output_path = f'docs/code-examples/{example}_rust.rrd'
    cmd = ['cargo', 'run', '--quiet', '-p', 'code_examples', '--bin', example]
    if target is not None:
        cmd += ['--target', target]
    if target_dir is not None:
        cmd += ['--target-dir', target_dir]
    if release:
        cmd += ['--release']
    if extra_args.get(example):
        cmd += ['--'] + extra_args[example]
    env = roundtrip_env(save_path=output_path)
    run(cmd, env=env, timeout=12000)
    return output_path

def run_roundtrip_cpp(example: str, release: bool) -> str:
    if False:
        print('Hello World!')
    target_name = f'{example}'
    output_path = f'docs/code-examples/{example}_cpp.rrd'
    cmake_build(target_name, release)
    cmd = [f'./build/docs/code-examples/{example}'] + (extra_args.get(example) or [])
    env = roundtrip_env(save_path=output_path)
    run(cmd, env=env, timeout=12000)
    return output_path

def cmake_build(target: str, release: bool) -> None:
    if False:
        return 10
    config = 'Debug'
    if release:
        config = 'Release'
    build_process_args = ['cmake', '--build', '.', '--config', config, '--target', target, '--parallel', str(multiprocessing.cpu_count())]
    run(build_process_args, cwd='build')

def run_comparison(rrd0_path: str, rrd1_path: str, full_dump: bool) -> None:
    if False:
        i = 10
        return i + 15
    cmd = ['rerun', 'compare']
    if full_dump:
        cmd += ['--full-dump']
    cmd += [rrd0_path, rrd1_path]
    run(cmd, env=roundtrip_env(), timeout=30)

def check_non_empty_rrd(path: str) -> None:
    if False:
        print('Hello World!')
    from pathlib import Path
    assert Path(path).stat().st_size > 0
if __name__ == '__main__':
    main()