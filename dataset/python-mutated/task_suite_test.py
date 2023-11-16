import pytest
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.testing.checklist_test import FakeTaskSuite

class TestTaskSuite(AllenNlpTestCase):

    def setup_method(self):
        if False:
            while True:
                i = 10
        super().setup_method()
        archive = load_archive(self.FIXTURES_ROOT / 'basic_classifier' / 'serialization' / 'model.tar.gz')
        self.predictor = Predictor.from_archive(archive)

    def test_load_from_suite_file(self):
        if False:
            return 10
        suite_file = str(self.FIXTURES_ROOT / 'task_suites' / 'fake_suite.tar.gz')
        task_suite = TaskSuite.constructor(suite_file=suite_file)
        assert len(task_suite.suite.tests) == 1

    def test_load_by_name(self):
        if False:
            print('Hello World!')
        task_suite = TaskSuite.constructor(name='fake-task-suite')
        assert task_suite._fake_arg1 is None
        assert task_suite._fake_arg2 is None
        assert len(task_suite.suite.tests) == 1
        with pytest.raises(ConfigurationError):
            TaskSuite.constructor(name='suite-that-does-not-exist')

    def test_load_with_extra_args(self):
        if False:
            for i in range(10):
                print('nop')
        extra_args = {'fake_arg1': 'some label'}
        task_suite = TaskSuite.constructor(name='fake-task-suite', extra_args=extra_args)
        assert task_suite._fake_arg1 == 'some label'

    def test_prediction_and_confidence_scores_function_needs_implementation(self):
        if False:
            return 10
        task_suite = TaskSuite.constructor(name='fake-task-suite')
        with pytest.raises(NotImplementedError):
            task_suite.run(self.predictor)

    def test_add_default_tests(self):
        if False:
            print('Hello World!')
        data = ["This isn't real data"]
        task_suite = TaskSuite(add_default_tests=True, data=data)
        assert 'Typos' in task_suite.suite.tests
        assert '2 Typos' in task_suite.suite.tests
        assert 'Contractions' in task_suite.suite.tests
        data = ['This is data with no contractions.']
        task_suite = TaskSuite(add_default_tests=True, data=data)
        assert 'Typos' in task_suite.suite.tests
        assert '2 Typos' in task_suite.suite.tests
        assert 'Contractions' not in task_suite.suite.tests