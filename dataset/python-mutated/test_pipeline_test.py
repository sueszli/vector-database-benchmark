"""Unit test for the TestPipeline class"""
import logging
import unittest
import mock
from hamcrest.core.assert_that import assert_that as hc_assert_that
from hamcrest.core.base_matcher import BaseMatcher
from apache_beam.internal import pickler
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing.test_pipeline import TestPipeline

class SimpleMatcher(BaseMatcher):

    def _matches(self, item):
        if False:
            while True:
                i = 10
        return True

class TestPipelineTest(unittest.TestCase):
    TEST_CASE = {'options': ['--test-pipeline-options', '--job=mockJob --male --age=1'], 'expected_list': ['--job=mockJob', '--male', '--age=1'], 'expected_dict': {'job': 'mockJob', 'male': True, 'age': 1}}

    class TestParsingOptions(PipelineOptions):

        @classmethod
        def _add_argparse_args(cls, parser):
            if False:
                print('Hello World!')
            parser.add_argument('--job', action='store', help='mock job')
            parser.add_argument('--male', action='store_true', help='mock gender')
            parser.add_argument('--age', action='store', type=int, help='mock age')

    def test_option_args_parsing(self):
        if False:
            for i in range(10):
                print('nop')
        test_pipeline = TestPipeline(argv=self.TEST_CASE['options'])
        self.assertListEqual(sorted(test_pipeline.get_full_options_as_args()), sorted(self.TEST_CASE['expected_list']))

    def test_empty_option_args_parsing(self):
        if False:
            for i in range(10):
                print('nop')
        test_pipeline = TestPipeline()
        self.assertListEqual([], test_pipeline.get_full_options_as_args())

    def test_create_test_pipeline_options(self):
        if False:
            i = 10
            return i + 15
        test_pipeline = TestPipeline(argv=self.TEST_CASE['options'])
        test_options = PipelineOptions(test_pipeline.get_full_options_as_args())
        self.assertDictContainsSubset(self.TEST_CASE['expected_dict'], test_options.get_all_options())
    EXTRA_OPT_CASES = [{'options': {'name': 'Mark'}, 'expected': ['--name=Mark']}, {'options': {'student': True}, 'expected': ['--student']}, {'options': {'student': False}, 'expected': []}, {'options': {'name': 'Mark', 'student': True}, 'expected': ['--name=Mark', '--student']}]

    def test_append_extra_options(self):
        if False:
            while True:
                i = 10
        test_pipeline = TestPipeline()
        for case in self.EXTRA_OPT_CASES:
            opt_list = test_pipeline.get_full_options_as_args(**case['options'])
            self.assertListEqual(sorted(opt_list), sorted(case['expected']))

    def test_append_verifier_in_extra_opt(self):
        if False:
            i = 10
            return i + 15
        extra_opt = {'matcher': SimpleMatcher()}
        opt_list = TestPipeline().get_full_options_as_args(**extra_opt)
        (_, value) = opt_list[0].split('=', 1)
        matcher = pickler.loads(value)
        self.assertTrue(isinstance(matcher, BaseMatcher))
        hc_assert_that(None, matcher)

    def test_get_option(self):
        if False:
            print('Hello World!')
        (name, value) = ('job', 'mockJob')
        test_pipeline = TestPipeline()
        test_pipeline.options_list = ['--%s=%s' % (name, value)]
        self.assertEqual(test_pipeline.get_option(name), value)

    def test_skip_IT(self):
        if False:
            for i in range(10):
                print('nop')
        with TestPipeline(is_integration_test=True) as _:
            pass
        self.fail()

    @mock.patch('apache_beam.testing.test_pipeline.Pipeline.run', autospec=True)
    def test_not_use_test_runner_api(self, mock_run):
        if False:
            while True:
                i = 10
        with TestPipeline(argv=['--not-use-test-runner-api'], blocking=False) as test_pipeline:
            pass
        mock_run.assert_called_once_with(test_pipeline, test_runner_api=False)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()