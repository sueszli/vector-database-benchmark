"""Tests to check that py_test are properly loaded in BUILD files."""
import os
import subprocess
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

def check_output_despite_error(args):
    if False:
        while True:
            i = 10
    'Get output of args from command line, even if there are errors.\n\n  Args:\n    args: a list of command line args.\n\n  Returns:\n    output as string.\n  '
    try:
        output = subprocess.check_output(args, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        output = e.output
    return output.strip()

def main():
    if False:
        for i in range(10):
            print('nop')
    try:
        targets = subprocess.check_output(['bazel', 'query', 'kind(py_test, //tensorflow/contrib/... + //tensorflow/python/... - //tensorflow/contrib/tensorboard/...)']).strip()
    except subprocess.CalledProcessError as e:
        targets = e.output
    targets = targets.decode('utf-8') if isinstance(targets, bytes) else targets
    valid_targets = []
    for target in targets.split('\n'):
        kind = check_output_despite_error(['buildozer', 'print kind', target])
        if kind == 'py_test':
            tags = check_output_despite_error(['buildozer', 'print tags', target])
            if 'no_pip' not in tags:
                valid_targets.append(target)
    build_files = set()
    for target in valid_targets:
        build_files.add(os.path.join(target[2:].split(':')[0], 'BUILD'))
    files_missing_load = []
    for build_file in build_files:
        updated_build_file = subprocess.check_output(['buildozer', '-stdout', 'new_load //tensorflow:tensorflow.bzl py_test', build_file])
        with open(build_file, 'r') as f:
            if f.read() != updated_build_file:
                files_missing_load.append(build_file)
    if files_missing_load:
        raise RuntimeError('The following files are missing %s:\n %s' % ('load("//tensorflow:tensorflow.bzl", "py_test").\nThis load statement is needed because otherwise pip tests will try to use their dependencies, which are not visible to them.', '\n'.join(files_missing_load)))
    else:
        print('TEST PASSED.')
if __name__ == '__main__':
    main()