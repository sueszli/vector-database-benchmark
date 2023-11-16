import pytest
import glob
import os
from os.path import join, expanduser, abspath
from mycroft.configuration import Configuration
from test.integrationtests.skills.skill_tester import MockSkillsLoader
from test.integrationtests.skills.skill_tester import SkillTest
from .runner import load_test_environment

def discover_tests(skills_dir):
    if False:
        while True:
            i = 10
    ' Find all tests for the skills in the default skill path,\n    or in the path provided as the LAST command line argument.\n\n    Finds intent test json files and corresponding .../test/__init__.py\n    containing a test_runner function allowing per skill mocking.\n\n    Returns:\n        Tests, lists of (intent example, test environment)\n    '
    tests = {}
    skills = [skill for skill in sorted(glob.glob(skills_dir + '/*')) if os.path.isdir(skill)]
    for skill in skills:
        test_env = load_test_environment(skill)
        test_intent_files = [(f, test_env) for f in sorted(glob.glob(os.path.join(skill, 'test/intent/*.json')))]
        if len(test_intent_files) > 0:
            tests[skill] = test_intent_files
    return tests

def get_skills_dir():
    if False:
        return 10
    return expanduser(os.environ.get('SKILLS_DIR', '')) or expanduser(join(Configuration.get()['data_dir'], Configuration.get()['skills']['msm']['directory']))

def run_test_setup(loader, tests):
    if False:
        return 10
    ' Run test_setup for all loaded skills. '
    for s in loader.skills:
        if len(tests.get(s.root_dir, [])) > 0:
            try:
                test_env = tests[s.root_dir][0]
                if hasattr(test_env[1], 'test_setup'):
                    print('Running test setup for {}'.format(s.name))
                    test_env[1].test_setup(s)
            except Exception as e:
                print('test_setup for {} failed: {}'.format(s.name, repr(e)))
skills_dir = get_skills_dir()
tests = discover_tests(skills_dir)
loader = MockSkillsLoader(skills_dir)
emitter = loader.load_skills()
skill_dir = os.environ.get('SKILL_DIR', '')
run_test_setup(loader, tests)

class TestCase(object):

    @pytest.mark.parametrize('skill,test', sum([[(skill, test) for test in tests[skill]] for skill in tests.keys() if not skill_dir or abspath(skill).startswith(abspath(skill_dir))], []))
    def test_skill(self, skill, test):
        if False:
            while True:
                i = 10
        (example, test_env) = test
        if test_env and hasattr(test_env, 'test_runner'):
            assert test_env.test_runner(skill, example, emitter, loader)
        else:
            t = SkillTest(skill, example, emitter)
            if not t.run(loader):
                assert False, 'Failure: ' + t.failure_msg