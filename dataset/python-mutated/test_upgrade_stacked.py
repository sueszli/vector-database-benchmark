"""Tests for upgrades of various stacking situations."""
from bzrlib import controldir, check, errors, tests
from bzrlib.upgrade import upgrade
from bzrlib.tests.scenarios import load_tests_apply_scenarios

def upgrade_scenarios():
    if False:
        for i in range(10):
            print('nop')
    scenario_pairs = [('knit', '1.6', False), ('1.6', '1.6.1-rich-root', True)]
    scenarios = []
    for (old_name, new_name, model_change) in scenario_pairs:
        name = old_name + ', ' + new_name
        scenarios.append((name, dict(scenario_old_format=old_name, scenario_new_format=new_name, scenario_model_change=model_change)))
    return scenarios
load_tests = load_tests_apply_scenarios

class TestStackUpgrade(tests.TestCaseWithTransport):
    scenarios = upgrade_scenarios()

    def test_stack_upgrade(self):
        if False:
            for i in range(10):
                print('nop')
        'Correct checks when stacked-on repository is upgraded.\n\n        We initially stack on a repo with the same rich root support,\n        we then upgrade it and should fail, we then upgrade the overlaid\n        repository.\n        '
        base = self.make_branch_and_tree('base', format=self.scenario_old_format)
        self.build_tree(['base/foo'])
        base.commit('base commit')
        stacked = base.bzrdir.sprout('stacked', stacked=True)
        self.assertTrue(stacked.open_branch().get_stacked_on_url())
        new_format = controldir.format_registry.make_bzrdir(self.scenario_new_format)
        upgrade('base', new_format)
        if self.scenario_model_change:
            self.assertRaises(errors.IncompatibleRepositories, stacked.open_branch)
        else:
            check.check_dwim('stacked', False, True, True)
        stacked = controldir.ControlDir.open('stacked')
        upgrade('stacked', new_format)
        stacked = controldir.ControlDir.open('stacked')
        check.check_dwim('stacked', False, True, True)