from subprocess import check_call, CalledProcessError
import argparse
import os
import logging
import sys
import json
from ci_tools.environment_exclusions import is_check_enabled, is_typing_ignored
from ci_tools.parsing import ParsedSetup
from ci_tools.variables import in_ci
from gh_tools.vnext_issue_creator import create_vnext_issue
logging.getLogger().setLevel(logging.INFO)
root_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))

def get_pyright_config_path(args):
    if False:
        for i in range(10):
            print('nop')
    "Give pyright an execution environment when running with tox. Otherwise\n    we use pyright's default for import resolution which doesn't work well\n    in our monorepo.\n\n    Since we don't want to be burdened with upkeep of this config for every library\n    that runs pyright checks, nor do we need this when running pyright without tox,\n    we'll add the config on the fly when invoked with tox.\n    "
    user_config_path = args.target_package if os.path.exists(os.path.join(args.target_package, 'pyrightconfig.json')) else root_dir
    with open(os.path.join(user_config_path, 'pyrightconfig.json'), 'r') as f:
        config_text = f.read()
        config_text = config_text.replace('"**', '"../../../../**')
        config = json.loads(config_text)
    if config.get('executionEnvironments'):
        config['executionEnvironments'].append({'root': args.target_package})
    else:
        config.update({'executionEnvironments': [{'root': args.target_package}]})
    if args.next:
        config['pythonVersion'] = '3.8'
    pyright_env = 'pyright' if not args.next else 'next-pyright'
    pyright_config_path = os.path.join(args.target_package, '.tox', pyright_env, 'tmp', 'pyrightconfig.json')
    with open(pyright_config_path, 'w+') as f:
        f.write(json.dumps(config, indent=4))
    return pyright_config_path
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pyright against target folder. ')
    parser.add_argument('-t', '--target', dest='target_package', help='The target package directory on disk. The target module passed to run pyright will be <target_package>/azure.', required=True)
    parser.add_argument('--next', default=False, help='Next version of pyright is being tested.', required=False)
    args = parser.parse_args()
    package_dir = os.path.abspath(args.target_package)
    package_name = os.path.basename(package_dir)
    if not args.next and in_ci():
        if not is_check_enabled(args.target_package, 'pyright') or is_typing_ignored(package_name):
            logging.info(f'Package {package_name} opts-out of pyright check. See https://aka.ms/python/typing-guide for information.')
            exit(0)
    pkg_details = ParsedSetup.from_path(package_dir)
    top_level_module = pkg_details.namespace.split('.')[0]
    paths = [os.path.join(args.target_package, top_level_module), os.path.join(args.target_package, 'samples')]
    if not args.next and in_ci():
        if not is_check_enabled(args.target_package, 'type_check_samples'):
            logging.info(f'Package {package_name} opts-out of pyright check on samples.')
            paths = paths[:-1]
    pyright_config_path = get_pyright_config_path(args)
    commands = [sys.executable, '-m', 'pyright', '--project', pyright_config_path]
    commands.extend(paths)
    try:
        check_call(commands)
    except CalledProcessError as error:
        if args.next and in_ci() and is_check_enabled(args.target_package, 'pyright') and (not is_typing_ignored(package_name)):
            create_vnext_issue(package_name, 'pyright')
        print('See https://aka.ms/python/typing-guide for information.\n\n')
        raise error