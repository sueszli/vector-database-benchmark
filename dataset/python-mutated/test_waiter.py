from __future__ import annotations
from typing import Any
from unittest import mock
from unittest.mock import MagicMock
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.amazon.aws.utils.waiter import waiter
SUCCESS_STATES = {'Created'}
FAILURE_STATES = {'Failed'}

def generate_response(state: str) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    return {'Status': {'State': state}}

def assert_expected_waiter_type(waiter: mock.MagicMock, expected: str):
    if False:
        return 10
    '\n    There does not appear to be a straight-forward way to assert the type of waiter.\n    Instead, get the class name and check if it contains the expected name.\n\n    :param waiter: A mocked Boto3 Waiter object.\n    :param expected: The expected class name of the Waiter object, for example "ClusterActive".\n    '
    assert expected in str(type(waiter.call_args.args[0]))

class TestWaiter:

    @pytest.mark.parametrize('get_state_responses, fails, expected_exception, expected_num_calls', [([generate_response('Created')], False, None, 1), ([generate_response('Failed')], True, AirflowException, 1), ([generate_response('Pending'), generate_response('Pending'), generate_response('Created')], False, None, 3), ([generate_response('Pending'), generate_response('Failed')], True, AirflowException, 2), ([generate_response('Pending'), generate_response('Pending'), generate_response('Failed')], True, AirflowException, 3), ([generate_response('Pending') for i in range(10)], True, RuntimeError, 5)])
    @mock.patch('time.sleep', return_value=None)
    def test_waiter(self, _, get_state_responses, fails, expected_exception, expected_num_calls):
        if False:
            for i in range(10):
                print('nop')
        mock_get_state = MagicMock()
        mock_get_state.side_effect = get_state_responses
        get_state_args = {}
        if fails:
            with pytest.raises(expected_exception):
                waiter(get_state_callable=mock_get_state, get_state_args=get_state_args, parse_response=['Status', 'State'], desired_state=SUCCESS_STATES, failure_states=FAILURE_STATES, object_type='test_object', action='testing', check_interval_seconds=1, countdown=5)
        else:
            waiter(get_state_callable=mock_get_state, get_state_args=get_state_args, parse_response=['Status', 'State'], desired_state=SUCCESS_STATES, failure_states=FAILURE_STATES, object_type='test_object', action='testing', check_interval_seconds=1, countdown=5)
        assert mock_get_state.call_count == expected_num_calls