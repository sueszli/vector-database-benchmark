"""
Convert the conda environment.yml to the pip requirements-dev.txt,
or check that they have the same packages (for the CI)

Usage:

    Generate `requirements-dev.txt`
    $ python scripts/generate_pip_deps_from_conda.py

    Compare and fail (exit status != 0) if `requirements-dev.txt` has not been
    generated with this script:
    $ python scripts/generate_pip_deps_from_conda.py --compare
"""
import argparse
import pathlib
import re
import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
import yaml
EXCLUDE = {'python', 'c-compiler', 'cxx-compiler'}
REMAP_VERSION = {'tzdata': '2022.7'}
RENAME = {'pytables': 'tables', 'psycopg2': 'psycopg2-binary', 'dask-core': 'dask', 'seaborn-base': 'seaborn', 'sqlalchemy': 'SQLAlchemy'}

def conda_package_to_pip(package: str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a conda package to its pip equivalent.\n\n    In most cases they are the same, those are the exceptions:\n    - Packages that should be excluded (in `EXCLUDE`)\n    - Packages that should be renamed (in `RENAME`)\n    - A package requiring a specific version, in conda is defined with a single\n      equal (e.g. ``pandas=1.0``) and in pip with two (e.g. ``pandas==1.0``)\n    '
    package = re.sub('(?<=[^<>])=', '==', package).strip()
    for compare in ('<=', '>=', '=='):
        if compare in package:
            (pkg, version) = package.split(compare)
            if pkg in EXCLUDE:
                return
            if pkg in REMAP_VERSION:
                return ''.join((pkg, compare, REMAP_VERSION[pkg]))
            if pkg in RENAME:
                return ''.join((RENAME[pkg], compare, version))
    if package in EXCLUDE:
        return
    if package in RENAME:
        return RENAME[package]
    return package

def generate_pip_from_conda(conda_path: pathlib.Path, pip_path: pathlib.Path, compare: bool=False) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Generate the pip dependencies file from the conda file, or compare that\n    they are synchronized (``compare=True``).\n\n    Parameters\n    ----------\n    conda_path : pathlib.Path\n        Path to the conda file with dependencies (e.g. `environment.yml`).\n    pip_path : pathlib.Path\n        Path to the pip file with dependencies (e.g. `requirements-dev.txt`).\n    compare : bool, default False\n        Whether to generate the pip file (``False``) or to compare if the\n        pip file has been generated with this script and the last version\n        of the conda file (``True``).\n\n    Returns\n    -------\n    bool\n        True if the comparison fails, False otherwise\n    '
    with conda_path.open() as file:
        deps = yaml.safe_load(file)['dependencies']
    pip_deps = []
    for dep in deps:
        if isinstance(dep, str):
            conda_dep = conda_package_to_pip(dep)
            if conda_dep:
                pip_deps.append(conda_dep)
        elif isinstance(dep, dict) and len(dep) == 1 and ('pip' in dep):
            pip_deps.extend(dep['pip'])
        else:
            raise ValueError(f'Unexpected dependency {dep}')
    header = f'# This file is auto-generated from {conda_path.name}, do not modify.\n# See that file for comments about the need/usage of each dependency.\n\n'
    pip_content = header + '\n'.join(pip_deps) + '\n'
    with open(pathlib.Path(conda_path.parent, 'pyproject.toml'), 'rb') as fd:
        meta = tomllib.load(fd)
    for requirement in meta['build-system']['requires']:
        if 'setuptools' in requirement:
            pip_content += requirement
            pip_content += '\n'
    if compare:
        with pip_path.open() as file:
            return pip_content != file.read()
    with pip_path.open('w') as file:
        file.write(pip_content)
    return False
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='convert (or compare) conda file to pip')
    argparser.add_argument('--compare', action='store_true', help='compare whether the two files are equivalent')
    args = argparser.parse_args()
    conda_fname = 'environment.yml'
    pip_fname = 'requirements-dev.txt'
    repo_path = pathlib.Path(__file__).parent.parent.absolute()
    res = generate_pip_from_conda(pathlib.Path(repo_path, conda_fname), pathlib.Path(repo_path, pip_fname), compare=args.compare)
    if res:
        msg = f'`{pip_fname}` has to be generated with `{__file__}` after `{conda_fname}` is modified.\n'
        sys.stderr.write(msg)
    sys.exit(res)