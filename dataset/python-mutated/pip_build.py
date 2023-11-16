"""Script to create (and optionally install) a `.whl` archive for Keras 3.

Usage:

1. Create a `.whl` file in `dist/`:

```
python3 pip_build.py
```

2. Also install the new package immediately after:

```
python3 pip_build.py --install
```
"""
import argparse
import datetime
import glob
import os
import pathlib
import shutil
import namex
import torch
package = 'keras'
build_directory = 'tmp_build_dir'
dist_directory = 'dist'
to_copy = ['setup.py', 'README.md']

def ignore_files(_, filenames):
    if False:
        for i in range(10):
            print('nop')
    return [f for f in filenames if f.endswith('_test.py')]

def copy_source_to_build_directory(root_path):
    if False:
        for i in range(10):
            print('nop')
    os.chdir(root_path)
    os.mkdir(build_directory)
    shutil.copytree(package, os.path.join(build_directory, package), ignore=ignore_files)
    for fname in to_copy:
        shutil.copy(fname, os.path.join(f'{build_directory}', fname))
    os.chdir(build_directory)

def run_namex_conversion():
    if False:
        i = 10
        return i + 15
    namex.convert_codebase(package, code_directory='src')
    namex.generate_api_files(package, code_directory='src', verbose=True)

def create_legacy_directory():
    if False:
        while True:
            i = 10
    tf_keras_dirpath_parent = os.path.join(package, '_tf_keras')
    tf_keras_dirpath = os.path.join(tf_keras_dirpath_parent, 'keras')
    os.makedirs(tf_keras_dirpath)
    with open(os.path.join(tf_keras_dirpath_parent, '__init__.py'), 'w') as f:
        f.write('from keras._tf_keras import keras\n')
    with open(os.path.join(package, '__init__.py')) as f:
        init_file = f.read()
        init_file = init_file.replace('from keras import _legacy', 'from keras import _tf_keras')
    with open(os.path.join(package, '__init__.py'), 'w') as f:
        f.write(init_file)
    with open(os.path.join(tf_keras_dirpath, '__init__.py'), 'w') as f:
        f.write(init_file)
    for dirname in os.listdir(package):
        dirpath = os.path.join(package, dirname)
        if os.path.isdir(dirpath) and dirname not in ('_legacy', '_tf_keras', 'src'):
            shutil.copytree(dirpath, os.path.join(tf_keras_dirpath, dirname), ignore=ignore_files)
    legacy_submodules = [path[:-3] for path in os.listdir(os.path.join(package, 'src', 'legacy')) if path.endswith('.py')]
    legacy_submodules += [path for path in os.listdir(os.path.join(package, 'src', 'legacy')) if os.path.isdir(os.path.join(package, 'src', 'legacy', path))]
    for (root, _, fnames) in os.walk(os.path.join(package, '_legacy')):
        for fname in fnames:
            if fname.endswith('.py'):
                legacy_fpath = os.path.join(root, fname)
                tf_keras_root = root.replace('/_legacy', '/_tf_keras/keras')
                core_api_fpath = os.path.join(root.replace('/_legacy', ''), fname)
                if not os.path.exists(tf_keras_root):
                    os.makedirs(tf_keras_root)
                tf_keras_fpath = os.path.join(tf_keras_root, fname)
                with open(legacy_fpath) as f:
                    legacy_contents = f.read()
                    legacy_contents = legacy_contents.replace('keras._legacy', 'keras._tf_keras.keras')
                if os.path.exists(core_api_fpath):
                    with open(core_api_fpath) as f:
                        core_api_contents = f.read()
                    core_api_contents = core_api_contents.replace('from keras import _tf_keras\n', '')
                    for legacy_submodule in legacy_submodules:
                        core_api_contents = core_api_contents.replace(f'from keras import {legacy_submodule}\n', '')
                        core_api_contents = core_api_contents.replace(f'keras.{legacy_submodule}', f'keras._tf_keras.keras.{legacy_submodule}')
                    legacy_contents = core_api_contents + '\n' + legacy_contents
                with open(tf_keras_fpath, 'w') as f:
                    f.write(legacy_contents)
    shutil.rmtree(os.path.join(package, '_legacy'))

def export_version_string(version, is_nightly=False, rc_index=None):
    if False:
        while True:
            i = 10
    'Export Version and Package Name.'
    if is_nightly:
        date = datetime.datetime.now()
        version += f".dev{date.strftime('%Y%m%d%H')}"
        with open('setup.py') as f:
            setup_contents = f.read()
        with open('setup.py', 'w') as f:
            setup_contents = setup_contents.replace('name="keras"', 'name="keras-nightly"')
            f.write(setup_contents)
    elif rc_index is not None:
        version += 'rc' + str(rc_index)
    with open(os.path.join(package, '__init__.py')) as f:
        init_contents = f.read()
    with open(os.path.join(package, '__init__.py'), 'w') as f:
        f.write(init_contents + '\n\n' + f'__version__ = "{version}"\n')

def build_and_save_output(root_path, __version__):
    if False:
        for i in range(10):
            print('nop')
    os.system('python3 -m build')
    os.chdir(root_path)
    if not os.path.exists(dist_directory):
        os.mkdir(dist_directory)
    for fpath in glob.glob(os.path.join(build_directory, dist_directory, '*.*')):
        shutil.copy(fpath, dist_directory)
    whl_path = None
    for fname in os.listdir(dist_directory):
        if __version__ in fname and fname.endswith('.whl'):
            whl_path = os.path.abspath(os.path.join(dist_directory, fname))
    if whl_path:
        print(f'Build successful. Wheel file available at {whl_path}')
    else:
        print('Build failed.')
    return whl_path

def build(root_path, is_nightly=False, rc_index=None):
    if False:
        while True:
            i = 10
    if os.path.exists(build_directory):
        raise ValueError(f'Directory already exists: {build_directory}')
    try:
        copy_source_to_build_directory(root_path)
        run_namex_conversion()
        create_legacy_directory()
        from keras.src.version import __version__
        export_version_string(__version__, is_nightly, rc_index)
        return build_and_save_output(root_path, __version__)
    finally:
        shutil.rmtree(build_directory)

def install_whl(whl_fpath):
    if False:
        print('Hello World!')
    print(f'Installing wheel file: {whl_fpath}')
    os.system(f'pip3 install {whl_fpath} --force-reinstall --no-dependencies')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', action='store_true', help='Whether to install the generated wheel file.')
    parser.add_argument('--nightly', action='store_true', help='Whether to generate nightly wheel file.')
    parser.add_argument('--rc', type=int, help='Specify `[0-9] when generating RC wheels.')
    args = parser.parse_args()
    root_path = pathlib.Path(__file__).parent.resolve()
    whl_path = build(root_path, args.nightly, args.rc)
    if whl_path and args.install:
        install_whl(whl_path)