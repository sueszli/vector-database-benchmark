import os
import sys
sys.path.append(os.path.dirname(__file__))
from mock.restful_server import init_response
from mock.experiment import create_mock_experiment, stop_mock_experiment, generate_args_parser, generate_args
from nni.tools.nnictl.nnictl_utils import get_experiment_time, get_experiment_status, check_experiment_id, parse_ids, get_config_filename, get_experiment_port, check_rest, trial_ls, list_experiment
import unittest
from unittest import TestCase, main
import responses

@unittest.skipIf(sys.platform == 'win32', 'Failed, debug later')
class CommonUtilsTestCase(TestCase):

    @classmethod
    def setUp(self):
        if False:
            i = 10
            return i + 15
        init_response()
        create_mock_experiment()

    @classmethod
    def tearDown(self):
        if False:
            i = 10
            return i + 15
        stop_mock_experiment()

    @responses.activate
    def test_get_experiment_status(self):
        if False:
            print('Hello World!')
        self.assertEqual('RUNNING', get_experiment_status(8080))

    @responses.activate
    def test_check_experiment_id(self):
        if False:
            return 10
        parser = generate_args_parser()
        args = parser.parse_args(['xOpEwA5w'])
        self.assertEqual('xOpEwA5w', check_experiment_id(args))

    @responses.activate
    def test_parse_ids(self):
        if False:
            return 10
        parser = generate_args_parser()
        args = parser.parse_args(['xOpEwA5w'])
        self.assertEqual(['xOpEwA5w'], parse_ids(args))

    @responses.activate
    def test_get_config_file_name(self):
        if False:
            while True:
                i = 10
        args = generate_args()
        self.assertEqual('xOpEwA5w', get_config_filename(args))

    @responses.activate
    def test_get_experiment_port(self):
        if False:
            print('Hello World!')
        args = generate_args()
        self.assertEqual(8080, get_experiment_port(args))

    @responses.activate
    def test_check_rest(self):
        if False:
            while True:
                i = 10
        args = generate_args()
        self.assertEqual(True, check_rest(args))

    @responses.activate
    def test_trial_ls(self):
        if False:
            print('Hello World!')
        args = generate_args()
        trials = trial_ls(args)
        self.assertEqual(trials[0]['id'], 'GPInz')
if __name__ == '__main__':
    main()