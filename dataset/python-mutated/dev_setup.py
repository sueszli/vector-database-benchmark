from __future__ import print_function
import sys
import glob
import os
import argparse
from collections import Counter
from subprocess import check_call, CalledProcessError
root_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..'))

def pip_command(command, additional_dir='.', error_ok=False):
    if False:
        while True:
            i = 10
    try:
        print('Executing: {} from {}'.format(command, additional_dir))
        check_call([sys.executable, '-m', 'pip'] + command.split(), cwd=os.path.join(root_dir, additional_dir))
        print()
    except CalledProcessError as err:
        print(err, file=sys.stderr)
        if not error_ok:
            sys.exit(1)

def select_install_type(pkg, run_develop, exceptions):
    if False:
        i = 10
        return i + 15
    argument = ''
    if run_develop:
        argument = '-e'
    if pkg in exceptions:
        if argument == '':
            argument = '-e'
        elif argument == '-e':
            argument = ''
    return argument
parser = argparse.ArgumentParser(description='Set up the dev environment for selected packages.')
parser.add_argument('--packageList', '-p', dest='packageList', default='', help='Comma separated list of targeted packages. Used to limit the number of packages that dependencies will be installed for.')
parser.add_argument('--disabledevelop', dest='install_in_develop_mode', default=True, action='store_false', help='Add this argument if you would prefer to install the package with a simple `pip install` versus `pip install -e`')
parser.add_argument('--exceptionlist', '-e', dest='exception_list', default='', help="Comma separated list of packages that we want to take the 'opposite' installation method for.")
args = parser.parse_args()
packages = {tuple(os.path.dirname(f).rsplit(os.sep, 1)) for f in glob.glob(os.path.join(root_dir, 'sdk/*/azure-*/setup.py')) + glob.glob(os.path.join(root_dir, 'tools/azure-*/setup.py'))}
packages = {package_name: base_folder for (base_folder, package_name) in packages}
exceptions = [p.strip() for p in args.exception_list.split(',')]
if not args.packageList:
    targeted_packages = list(packages.keys())
else:
    targeted_packages = [os.path.relpath(x.strip()) for x in args.packageList.split(',')]
nspkg_packages = [p for p in packages.keys() if 'nspkg' in p]
nspkg_packages.sort(key=lambda x: len([c for c in x if c == '-']))
meta_packages = ['azure-keyvault', 'azure-mgmt', 'azure']
content_packages = sorted([p for p in packages.keys() if p not in nspkg_packages + meta_packages and p in targeted_packages])
if 'azure-devtools' in content_packages:
    content_packages.remove('azure-devtools')
content_packages.insert(0, 'azure-devtools')
if 'azure-sdk-tools' in content_packages:
    content_packages.remove('azure-sdk-tools')
content_packages.insert(1, 'azure-sdk-tools')
if 'azure-common' in content_packages:
    content_packages.remove('azure-common')
content_packages.insert(2, 'azure-common')
if 'azure-core' in content_packages:
    content_packages.remove('azure-core')
content_packages.insert(3, 'azure-core')
print('Running dev setup...')
print("Root directory '{}'\n".format(root_dir))
privates_dir = os.path.join(root_dir, 'privates')
if os.path.isdir(privates_dir) and os.listdir(privates_dir):
    whl_list = ' '.join([os.path.join(privates_dir, f) for f in os.listdir(privates_dir)])
    pip_command('install {}'.format(whl_list))
if sys.version_info < (3,):
    for package_name in nspkg_packages:
        pip_command('install {}/{}/'.format(packages[package_name], package_name))
print('Packages to install: {}'.format(content_packages))
for package_name in content_packages:
    print('\nInstalling {}'.format(package_name))
    if os.path.isfile('{}/{}/dev_requirements.txt'.format(packages[package_name], package_name)):
        pip_command('install -r dev_requirements.txt', os.path.join(packages[package_name], package_name))
    pip_command('install --ignore-requires-python {} {}'.format(select_install_type(package_name, args.install_in_develop_mode, exceptions), os.path.join(packages[package_name], package_name)))
if sys.version_info >= (3,):
    pip_command('uninstall -y azure-nspkg', error_ok=True)
print('Finished dev setup.')