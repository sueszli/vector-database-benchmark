"""This script allows you to develop Ray Python code without needing to compile
Ray.
See https://docs.ray.io/en/master/development.html#building-ray-python-only"""
import os
import sys
this_dir = os.path.dirname(__file__)
if this_dir in sys.path:
    cur = sys.path.remove(this_dir)
    sys.path.append(this_dir)
import argparse
import click
import shutil
import subprocess
import ray

def do_link(package, force=False, skip_list=None, local_path=None):
    if False:
        i = 10
        return i + 15
    if skip_list and package in skip_list:
        print(f'Skip creating symbolic link for {package}')
        return
    package_home = os.path.abspath(os.path.join(ray.__file__, f'../{package}'))
    if local_path is None:
        local_path = f'../{package}'
    local_home = os.path.abspath(os.path.join(__file__, local_path))
    if not os.path.isdir(package_home) and (not os.path.isfile(package_home)):
        print(f'{package_home} does not exist. Continuing to link.')
    assert os.path.exists(local_home), local_home
    if not force and (not click.confirm(f'This will replace:\n  {package_home}\nwith a symlink to:\n  {local_home}', default=True)):
        return
    if os.name == 'nt':
        try:
            shutil.rmtree(package_home)
        except FileNotFoundError:
            pass
        except OSError:
            os.remove(package_home)
        if os.path.isdir(local_home):
            subprocess.check_call(['mklink', '/J', package_home, local_home], shell=True)
        elif os.path.isfile(local_home):
            subprocess.check_call(['mklink', '/H', package_home, local_home], shell=True)
        else:
            print(f'{local_home} is neither directory nor file. Link failed.')
    else:
        sudo = []
        if not os.access(os.path.dirname(package_home), os.W_OK):
            print(f"You don't have write permission to {package_home}, using sudo:")
            sudo = ['sudo']
        print(f'Creating symbolic link from \n {local_home} to \n {package_home}')
        subprocess.check_call(sudo + ['rm', '-rf', package_home])
        subprocess.check_call(sudo + ['ln', '-s', local_home, package_home])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='Setup dev.')
    parser.add_argument('--yes', '-y', action='store_true', help="Don't ask for confirmation.")
    parser.add_argument('--skip', '-s', nargs='*', help='List of folders to skip linking to facilitate workspace dev', required=False)
    parser.add_argument('--extras', '-e', nargs='*', help='List of extra folders to link to facilitate workspace dev', required=False)
    args = parser.parse_args()
    if not args.yes:
        print("NOTE: Use '-y' to override all python files without confirmation.")
    do_link('rllib', force=args.yes, skip_list=args.skip, local_path='../../../rllib')
    do_link('air', force=args.yes, skip_list=args.skip)
    do_link('tune', force=args.yes, skip_list=args.skip)
    do_link('train', force=args.yes, skip_list=args.skip)
    do_link('autoscaler', force=args.yes, skip_list=args.skip)
    do_link('cloudpickle', force=args.yes, skip_list=args.skip)
    do_link('data', force=args.yes, skip_list=args.skip)
    do_link('scripts', force=args.yes, skip_list=args.skip)
    do_link('internal', force=args.yes, skip_list=args.skip)
    do_link('tests', force=args.yes, skip_list=args.skip)
    do_link('experimental', force=args.yes, skip_list=args.skip)
    do_link('util', force=args.yes, skip_list=args.skip)
    do_link('workflow', force=args.yes, skip_list=args.skip)
    do_link('serve', force=args.yes, skip_list=args.skip)
    do_link('dag', force=args.yes, skip_list=args.skip)
    do_link('widgets', force=args.yes, skip_list=args.skip)
    do_link('cluster_utils.py', force=args.yes, skip_list=args.skip)
    do_link('_private', force=args.yes, skip_list=args.skip)
    do_link('dashboard', force=args.yes, skip_list=args.skip, local_path='../../../dashboard')
    if args.extras is not None:
        for package in args.extras:
            do_link(package, force=args.yes, skip_list=args.skip)
    print('Created links.\n\nIf you run into issues initializing Ray, please ensure that your local repo and the installed Ray are in sync (pip install -U the latest wheels at https://docs.ray.io/en/master/installation.html, and ensure you are up-to-date on the master branch on git).\n\nNote that you may need to delete the package symlinks when pip installing new Ray versions to prevent pip from overwriting files in your git repo.')