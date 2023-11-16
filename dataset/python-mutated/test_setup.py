import argparse
from argparse import RawTextHelpFormatter
from glob import glob
from os.path import join, dirname, expanduser, basename, exists
from random import shuffle
import shutil
import sys
import yaml
from msm import MycroftSkillsManager, SkillRepo
from msm.exceptions import MsmException
from .generate_feature import generate_feature
'Test environment setup for voigt kampff test\n\nThe script sets up the selected tests in the feature directory so they can\nbe found and executed by the behave framework.\n\nThe script also ensures that the skills marked for testing are installed and\nthat any specified extra skills also gets installed into the environment.\n'
FEATURE_DIR = join(dirname(__file__), 'features') + '/'

def copy_config_definition_files(source, destination):
    if False:
        while True:
            i = 10
    'Copy all feature files from source to destination.'
    for f in glob(join(source, '*.config.json')):
        shutil.copyfile(f, join(destination, basename(f)))

def copy_feature_files(source, destination):
    if False:
        for i in range(10):
            print('nop')
    'Copy all feature files from source to destination.'
    for f in glob(join(source, '*.feature')):
        shutil.copyfile(f, join(destination, basename(f)))

def copy_step_files(source, destination):
    if False:
        print('Hello World!')
    'Copy all python files from source to destination.'
    for f in glob(join(source, '*.py')):
        shutil.copyfile(f, join(destination, basename(f)))

def apply_config(config, args):
    if False:
        i = 10
        return i + 15
    'Load config and add to unset arguments.'
    with open(expanduser(config)) as f:
        conf_dict = yaml.safe_load(f)
    if not args.test_skills and 'test_skills' in conf_dict:
        args.test_skills = conf_dict['test_skills']
    if not args.extra_skills and 'extra_skills' in conf_dict:
        args.extra_skills = conf_dict['extra_skills']
    if not args.platform and 'platform' in conf_dict:
        args.platform = conf_dict['platform']

def create_argument_parser():
    if False:
        i = 10
        return i + 15
    'Create the argument parser for the command line options.\n\n    Returns: ArgumentParser\n    '

    class TestSkillAction(argparse.Action):

        def __call__(self, parser, args, values, option_string=None):
            if False:
                print('Hello World!')
            args.test_skills = values.replace(',', ' ').split()

    class ExtraSkillAction(argparse.Action):

        def __call__(self, parser, args, values, option_string=None):
            if False:
                print('Hello World!')
            args.extra_skills = values.replace(',', ' ').split()
    platforms = list(MycroftSkillsManager.SKILL_GROUPS)
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-p', '--platform', choices=platforms, default='mycroft_mark_1')
    parser.add_argument('-t', '--test-skills', default=[], action=TestSkillAction, help='Comma-separated list of skills to test.\nEx: "mycroft-weather, mycroft-stock"')
    parser.add_argument('-s', '--extra-skills', default=[], action=ExtraSkillAction, help='Comma-separated list of extra skills to install.\nEx: "cocktails, laugh"')
    parser.add_argument('-r', '--random-skills', default=0, type=int, help='Number of random skills to install.')
    parser.add_argument('-d', '--skills-dir')
    parser.add_argument('-u', '--repo-url', help='URL for skills repo to install / update from')
    parser.add_argument('-b', '--branch', help='repo branch to use')
    parser.add_argument('-c', '--config', help='Path to test configuration file.')
    return parser

def get_random_skills(msm, num_random_skills):
    if False:
        while True:
            i = 10
    'Install random skills from uninstalled skill list.'
    random_skills = [s for s in msm.all_skills if not s.is_local]
    shuffle(random_skills)
    return [s.name for s in random_skills[:num_random_skills]]

def install_or_upgrade_skills(msm, skills):
    if False:
        return 10
    'Install needed skills if uninstalled, otherwise try to update.\n\n    Args:\n        msm: msm instance to use for the operations\n        skills: list of skills\n    '
    skills = [msm.find_skill(s) for s in skills]
    for s in skills:
        if not s.is_local:
            print('Installing {}'.format(s))
            msm.install(s)
        else:
            try:
                msm.update(s)
            except MsmException:
                pass

def collect_test_cases(msm, skills):
    if False:
        return 10
    'Collect feature files and step files for each skill.\n\n    Args:\n        msm: msm instance to use for the operations\n        skills: list of skills\n    '
    for skill_name in skills:
        skill = msm.find_skill(skill_name)
        behave_dir = join(skill.path, 'test', 'behave')
        if exists(behave_dir):
            copy_feature_files(behave_dir, FEATURE_DIR)
            copy_config_definition_files(behave_dir, FEATURE_DIR)
            step_dir = join(behave_dir, 'steps')
            if exists(step_dir):
                copy_step_files(step_dir, join(FEATURE_DIR, 'steps'))
        else:
            print('No feature files exists for {}, generating...'.format(skill_name))
            feature = generate_feature(skill_name, skill.path)
            with open(join(FEATURE_DIR, skill_name + '.feature'), 'w') as f:
                f.write(feature)

def print_install_report(platform, test_skills, extra_skills):
    if False:
        return 10
    'Print in nice format.'
    print('-------- TEST SETUP --------')
    yml = yaml.dump({'platform': platform, 'test_skills': test_skills, 'extra_skills': extra_skills})
    print(yml)
    print('----------------------------')

def get_arguments(cmdline_args):
    if False:
        return 10
    'Get arguments for test setup.\n\n    Parses the commandline and if specified applies configuration file.\n\n    Args:\n        cmdline_args (list): argv like list of arguments\n\n    Returns:\n        Argument parser NameSpace\n    '
    parser = create_argument_parser()
    args = parser.parse_args(cmdline_args)
    return args

def create_skills_manager(platform, skills_dir, url, branch):
    if False:
        return 10
    'Create mycroft skills manager for the given url / branch.\n\n    Args:\n        platform (str): platform to use\n        skills_dir (str): skill directory to use\n        url (str): skills repo url\n        branch (str): skills repo branch\n\n    Returns:\n        MycroftSkillsManager\n    '
    repo = SkillRepo(url=url, branch=branch)
    return MycroftSkillsManager(platform, skills_dir, repo)

def main(args):
    if False:
        print('Hello World!')
    'Parse arguments and run test environment setup.\n\n    This installs and/or upgrades any skills needed for the tests and\n    collects the feature and step files for the skills.\n    '
    if args.config:
        apply_config(args.config, args)
    msm = create_skills_manager(args.platform, args.skills_dir, args.repo_url, args.branch)
    random_skills = get_random_skills(msm, args.random_skills)
    all_skills = args.test_skills + args.extra_skills + random_skills
    install_or_upgrade_skills(msm, all_skills)
    collect_test_cases(msm, args.test_skills)
    print_install_report(msm.platform, args.test_skills, args.extra_skills + random_skills)
if __name__ == '__main__':
    main(get_arguments(sys.argv[1:]))