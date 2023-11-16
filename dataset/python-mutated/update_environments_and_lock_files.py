"""Script to update CI environment files and associated lock files.

To run it you need to be in the root folder of the scikit-learn repo:
python build_tools/update_environments_and_lock_files.py

Two scenarios where this script can be useful:
- make sure that the latest versions of all the dependencies are used in the CI.
  We can run this script regularly and open a PR with the changes to the lock
  files. This workflow will eventually be automated with a bot in the future.
- bump minimum dependencies in sklearn/_min_dependencies.py. Running this
  script will update both the CI environment files and associated lock files.
  You can then open a PR with the changes.
- pin some packages to an older version by adding them to the
  default_package_constraints variable. This is useful when regressions are
  introduced in our dependencies, this has happened for example with pytest 7
  and coverage 6.3.

Environments are conda environment.yml or pip requirements.txt. Lock files are
conda-lock lock files or pip-compile requirements.txt.

pip requirements.txt are used when we install some dependencies (e.g. numpy and
scipy) with apt-get and the rest of the dependencies (e.g. pytest and joblib)
with pip.

To run this script you need:
- conda-lock. The version should match the one used in the CI in
  sklearn/_min_dependencies.py
- pip-tools

To only update the environment and lock files for specific builds, you can use
the command line argument `--select-build` which will take a regex. For example,
to only update the documentation builds you can use:
`python build_tools/update_environments_and_lock_files.py --select-build doc`
"""
import json
import logging
import re
import shlex
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
import click
from jinja2 import Environment
from packaging.version import Version
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)
TRACE = logging.DEBUG - 5
common_dependencies_without_coverage = ['python', 'numpy', 'blas', 'scipy', 'cython', 'joblib', 'threadpoolctl', 'matplotlib', 'pandas', 'pyamg', 'pytest', 'pytest-xdist', 'pillow', 'setuptools']
common_dependencies = common_dependencies_without_coverage + ['pytest-cov', 'coverage']
docstring_test_dependencies = ['sphinx', 'numpydoc']
default_package_constraints = {'pytest-xdist': '2.5.0'}

def remove_from(alist, to_remove):
    if False:
        i = 10
        return i + 15
    return [each for each in alist if each not in to_remove]
conda_build_metadata_list = [{'build_name': 'pylatest_conda_forge_mkl_linux-64', 'folder': 'build_tools/azure', 'platform': 'linux-64', 'channel': 'conda-forge', 'conda_dependencies': common_dependencies + ['ccache', 'pytorch', 'pytorch-cpu', 'polars', 'pyarrow', 'array-api-compat'], 'package_constraints': {'blas': '[build=mkl]', 'pytorch': '1.13'}}, {'build_name': 'pylatest_conda_forge_mkl_osx-64', 'folder': 'build_tools/azure', 'platform': 'osx-64', 'channel': 'conda-forge', 'conda_dependencies': common_dependencies + ['ccache', 'compilers', 'llvm-openmp'], 'package_constraints': {'blas': '[build=mkl]'}}, {'build_name': 'pylatest_conda_mkl_no_openmp', 'folder': 'build_tools/azure', 'platform': 'osx-64', 'channel': 'defaults', 'conda_dependencies': common_dependencies + ['ccache'], 'package_constraints': {'blas': '[build=mkl]', 'numpy': '<1.25'}}, {'build_name': 'pylatest_conda_forge_mkl_no_coverage', 'folder': 'build_tools/azure', 'platform': 'linux-64', 'channel': 'conda-forge', 'conda_dependencies': common_dependencies_without_coverage + ['ccache'], 'package_constraints': {'blas': '[build=mkl]'}}, {'build_name': 'py38_conda_defaults_openblas', 'folder': 'build_tools/azure', 'platform': 'linux-64', 'channel': 'defaults', 'conda_dependencies': common_dependencies + ['ccache'], 'package_constraints': {'python': '3.8', 'blas': '[build=openblas]', 'numpy': 'min', 'scipy': 'min', 'matplotlib': 'min', 'threadpoolctl': '2.2.0', 'cython': 'min'}}, {'build_name': 'py38_conda_forge_openblas_ubuntu_2204', 'folder': 'build_tools/azure', 'platform': 'linux-64', 'channel': 'conda-forge', 'conda_dependencies': common_dependencies_without_coverage + ['ccache'], 'package_constraints': {'python': '3.8', 'blas': '[build=openblas]'}}, {'build_name': 'pylatest_pip_openblas_pandas', 'folder': 'build_tools/azure', 'platform': 'linux-64', 'channel': 'defaults', 'conda_dependencies': ['python', 'ccache'], 'pip_dependencies': remove_from(common_dependencies, ['python', 'blas']) + docstring_test_dependencies + ['lightgbm', 'scikit-image'], 'package_constraints': {'python': '3.9'}}, {'build_name': 'pylatest_pip_scipy_dev', 'folder': 'build_tools/azure', 'platform': 'linux-64', 'channel': 'defaults', 'conda_dependencies': ['python', 'ccache'], 'pip_dependencies': remove_from(common_dependencies, ['python', 'blas', 'matplotlib', 'pyamg', 'numpy', 'scipy', 'pandas', 'cython', 'joblib', 'pillow']) + ['pooch'] + docstring_test_dependencies + ['python-dateutil'], 'package_constraints': {'python': '3.11'}}, {'build_name': 'pypy3', 'folder': 'build_tools/azure', 'platform': 'linux-64', 'channel': 'conda-forge', 'conda_dependencies': ['pypy', 'python'] + remove_from(common_dependencies_without_coverage, ['python', 'pandas', 'pillow']) + ['ccache'], 'package_constraints': {'blas': '[build=openblas]', 'python': '3.9'}}, {'build_name': 'py38_conda_forge_mkl', 'folder': 'build_tools/azure', 'platform': 'win-64', 'channel': 'conda-forge', 'conda_dependencies': remove_from(common_dependencies, ['pandas', 'pyamg']) + ['wheel', 'pip'], 'package_constraints': {'python': '3.8', 'blas': '[build=mkl]'}}, {'build_name': 'doc_min_dependencies', 'folder': 'build_tools/circle', 'platform': 'linux-64', 'channel': 'conda-forge', 'conda_dependencies': common_dependencies_without_coverage + ['scikit-image', 'seaborn', 'memory_profiler', 'compilers', 'sphinx', 'sphinx-gallery', 'sphinx-copybutton', 'numpydoc', 'sphinx-prompt', 'plotly', 'pooch'], 'pip_dependencies': ['sphinxext-opengraph'], 'package_constraints': {'python': '3.8', 'numpy': 'min', 'scipy': 'min', 'matplotlib': 'min', 'cython': 'min', 'scikit-image': 'min', 'sphinx': 'min', 'pandas': 'min', 'sphinx-gallery': 'min', 'sphinx-copybutton': 'min', 'numpydoc': 'min', 'sphinx-prompt': 'min', 'sphinxext-opengraph': 'min', 'plotly': 'min'}}, {'build_name': 'doc', 'folder': 'build_tools/circle', 'platform': 'linux-64', 'channel': 'conda-forge', 'conda_dependencies': common_dependencies_without_coverage + ['scikit-image', 'seaborn', 'memory_profiler', 'compilers', 'sphinx', 'sphinx-gallery', 'sphinx-copybutton', 'numpydoc', 'sphinx-prompt', 'plotly', 'pooch', 'sphinxext-opengraph'], 'pip_dependencies': ['jupyterlite-sphinx', 'jupyterlite-pyodide-kernel'], 'package_constraints': {'python': '3.9'}}, {'build_name': 'py39_conda_forge', 'folder': 'build_tools/cirrus', 'platform': 'linux-aarch64', 'channel': 'conda-forge', 'conda_dependencies': remove_from(common_dependencies_without_coverage, ['pandas', 'pyamg']) + ['pip', 'ccache'], 'package_constraints': {'python': '3.9'}}]
pip_build_metadata_list = [{'build_name': 'debian_atlas_32bit', 'folder': 'build_tools/azure', 'pip_dependencies': ['cython', 'joblib', 'threadpoolctl', 'pytest', 'pytest-cov'], 'package_constraints': {'joblib': 'min', 'threadpoolctl': '2.2.0', 'pytest': 'min', 'pytest-cov': 'min', 'cython': 'min'}, 'python_version': '3.9.2'}, {'build_name': 'ubuntu_atlas', 'folder': 'build_tools/azure', 'pip_dependencies': ['cython', 'joblib', 'threadpoolctl', 'pytest', 'pytest-xdist'], 'package_constraints': {'joblib': 'min', 'threadpoolctl': 'min', 'cython': 'min'}, 'python_version': '3.10.4'}]

def execute_command(command_list):
    if False:
        return 10
    logger.debug(' '.join(command_list))
    proc = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    (out, err) = (out.decode(), err.decode())
    if proc.returncode != 0:
        command_str = ' '.join(command_list)
        raise RuntimeError('Command exited with non-zero exit code.\nExit code: {}\nCommand:\n{}\nstdout:\n{}\nstderr:\n{}\n'.format(proc.returncode, command_str, out, err))
    logger.log(TRACE, out)
    return out

def get_package_with_constraint(package_name, build_metadata, uses_pip=False):
    if False:
        print('Hello World!')
    build_package_constraints = build_metadata.get('package_constraints')
    if build_package_constraints is None:
        constraint = None
    else:
        constraint = build_package_constraints.get(package_name)
    constraint = constraint or default_package_constraints.get(package_name)
    if constraint is None:
        return package_name
    comment = ''
    if constraint == 'min':
        constraint = execute_command([sys.executable, 'sklearn/_min_dependencies.py', package_name]).strip()
        comment = '  # min'
    if re.match('\\d[.\\d]*', constraint):
        equality = '==' if uses_pip else '='
        constraint = equality + constraint
    return f'{package_name}{constraint}{comment}'
environment = Environment(trim_blocks=True, lstrip_blocks=True)
environment.filters['get_package_with_constraint'] = get_package_with_constraint

def get_conda_environment_content(build_metadata):
    if False:
        while True:
            i = 10
    template = environment.from_string("\n# DO NOT EDIT: this file is generated from the specification found in the\n# following script to centralize the configuration for CI builds:\n# build_tools/update_environments_and_lock_files.py\nchannels:\n  - {{ build_metadata['channel'] }}\ndependencies:\n  {% for conda_dep in build_metadata['conda_dependencies'] %}\n  - {{ conda_dep | get_package_with_constraint(build_metadata) }}\n  {% endfor %}\n  {% if build_metadata['pip_dependencies'] %}\n  - pip\n  - pip:\n  {% for pip_dep in build_metadata.get('pip_dependencies', []) %}\n    - {{ pip_dep | get_package_with_constraint(build_metadata, uses_pip=True) }}\n  {% endfor %}\n  {% endif %}".strip())
    return template.render(build_metadata=build_metadata)

def write_conda_environment(build_metadata):
    if False:
        return 10
    content = get_conda_environment_content(build_metadata)
    build_name = build_metadata['build_name']
    folder_path = Path(build_metadata['folder'])
    output_path = folder_path / f'{build_name}_environment.yml'
    logger.debug(output_path)
    output_path.write_text(content)

def write_all_conda_environments(build_metadata_list):
    if False:
        for i in range(10):
            print('nop')
    for build_metadata in build_metadata_list:
        write_conda_environment(build_metadata)

def conda_lock(environment_path, lock_file_path, platform):
    if False:
        i = 10
        return i + 15
    command = f'conda-lock lock --mamba --kind explicit --platform {platform} --file {environment_path} --filename-template {lock_file_path}'
    execute_command(shlex.split(command))

def create_conda_lock_file(build_metadata):
    if False:
        print('Hello World!')
    build_name = build_metadata['build_name']
    folder_path = Path(build_metadata['folder'])
    environment_path = folder_path / f'{build_name}_environment.yml'
    platform = build_metadata['platform']
    lock_file_basename = build_name
    if not lock_file_basename.endswith(platform):
        lock_file_basename = f'{lock_file_basename}_{platform}'
    lock_file_path = folder_path / f'{lock_file_basename}_conda.lock'
    conda_lock(environment_path, lock_file_path, platform)

def write_all_conda_lock_files(build_metadata_list):
    if False:
        for i in range(10):
            print('nop')
    for build_metadata in build_metadata_list:
        logger.info(f"# Locking dependencies for {build_metadata['build_name']}")
        create_conda_lock_file(build_metadata)

def get_pip_requirements_content(build_metadata):
    if False:
        i = 10
        return i + 15
    template = environment.from_string("\n# DO NOT EDIT: this file is generated from the specification found in the\n# following script to centralize the configuration for CI builds:\n# build_tools/update_environments_and_lock_files.py\n{% for pip_dep in build_metadata['pip_dependencies'] %}\n{{ pip_dep | get_package_with_constraint(build_metadata, uses_pip=True) }}\n{% endfor %}".strip())
    return template.render(build_metadata=build_metadata)

def write_pip_requirements(build_metadata):
    if False:
        print('Hello World!')
    build_name = build_metadata['build_name']
    content = get_pip_requirements_content(build_metadata)
    folder_path = Path(build_metadata['folder'])
    output_path = folder_path / f'{build_name}_requirements.txt'
    logger.debug(output_path)
    output_path.write_text(content)

def write_all_pip_requirements(build_metadata_list):
    if False:
        print('Hello World!')
    for build_metadata in build_metadata_list:
        write_pip_requirements(build_metadata)

def pip_compile(pip_compile_path, requirements_path, lock_file_path):
    if False:
        for i in range(10):
            print('nop')
    command = f'{pip_compile_path} --upgrade {requirements_path} -o {lock_file_path}'
    execute_command(shlex.split(command))

def write_pip_lock_file(build_metadata):
    if False:
        for i in range(10):
            print('nop')
    build_name = build_metadata['build_name']
    python_version = build_metadata['python_version']
    environment_name = f'pip-tools-python{python_version}'
    command = f'conda create -c conda-forge -n pip-tools-python{python_version} python={python_version} pip-tools -y'
    execute_command(shlex.split(command))
    json_output = execute_command(shlex.split('conda info --json'))
    conda_info = json.loads(json_output)
    environment_folder = [each for each in conda_info['envs'] if each.endswith(environment_name)][0]
    environment_path = Path(environment_folder)
    pip_compile_path = environment_path / 'bin' / 'pip-compile'
    folder_path = Path(build_metadata['folder'])
    requirement_path = folder_path / f'{build_name}_requirements.txt'
    lock_file_path = folder_path / f'{build_name}_lock.txt'
    pip_compile(pip_compile_path, requirement_path, lock_file_path)

def write_all_pip_lock_files(build_metadata_list):
    if False:
        while True:
            i = 10
    for build_metadata in build_metadata_list:
        logger.info(f"# Locking dependencies for {build_metadata['build_name']}")
        write_pip_lock_file(build_metadata)

def check_conda_lock_version():
    if False:
        return 10
    expected_conda_lock_version = execute_command([sys.executable, 'sklearn/_min_dependencies.py', 'conda-lock']).strip()
    installed_conda_lock_version = version('conda-lock')
    if installed_conda_lock_version != expected_conda_lock_version:
        raise RuntimeError(f'Expected conda-lock version: {expected_conda_lock_version}, got: {installed_conda_lock_version}')

def check_conda_version():
    if False:
        return 10
    conda_info_output = execute_command(['conda', 'info', '--json'])
    conda_info = json.loads(conda_info_output)
    conda_version = Version(conda_info['conda_version'])
    if Version('22.9.0') < conda_version < Version('23.7'):
        raise RuntimeError(f'conda version should be <= 22.9.0 or >= 23.7 got: {conda_version}')

@click.command()
@click.option('--select-build', default='', help='Regex to restrict the builds we want to update environment and lock files. By default all the builds are selected.')
@click.option('--skip-build', default=None, help='Regex to skip some builds from the builds selected by --select-build')
@click.option('-v', '--verbose', is_flag=True, help='Print commands executed by the script')
@click.option('-vv', '--very-verbose', is_flag=True, help='Print output of commands executed by the script')
def main(verbose, very_verbose, select_build, skip_build):
    if False:
        for i in range(10):
            print('nop')
    if verbose:
        logger.setLevel(logging.DEBUG)
    if very_verbose:
        logger.setLevel(TRACE)
        handler.setLevel(TRACE)
    check_conda_lock_version()
    check_conda_version()
    filtered_conda_build_metadata_list = [each for each in conda_build_metadata_list if re.search(select_build, each['build_name'])]
    if skip_build is not None:
        filtered_conda_build_metadata_list = [each for each in filtered_conda_build_metadata_list if not re.search(skip_build, each['build_name'])]
    if filtered_conda_build_metadata_list:
        logger.info('# Writing conda environments')
        write_all_conda_environments(filtered_conda_build_metadata_list)
        logger.info('# Writing conda lock files')
        write_all_conda_lock_files(filtered_conda_build_metadata_list)
    filtered_pip_build_metadata_list = [each for each in pip_build_metadata_list if re.search(select_build, each['build_name'])]
    if skip_build is not None:
        filtered_pip_build_metadata_list = [each for each in filtered_pip_build_metadata_list if not re.search(skip_build, each['build_name'])]
    if filtered_pip_build_metadata_list:
        logger.info('# Writing pip requirements')
        write_all_pip_requirements(filtered_pip_build_metadata_list)
        logger.info('# Writing pip lock files')
        write_all_pip_lock_files(filtered_pip_build_metadata_list)
if __name__ == '__main__':
    main()