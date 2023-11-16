from unittest.mock import MagicMock
import pytest
from rest_framework.request import Request
from organisations.subscriptions.decorators import require_plan
from organisations.subscriptions.exceptions import InvalidSubscriptionPlanError

def test_require_plan_raises_exception_if_plan_invalid():
    if False:
        i = 10
        return i + 15
    valid_plan_id = 'plan-id'
    invalid_plan_id = 'invalid-plan-id'
    mock_request = MagicMock(spec=Request)
    mock_subscription = MagicMock(plan=invalid_plan_id)

    @require_plan([valid_plan_id], lambda v: mock_subscription)
    def test_function(request: Request):
        if False:
            i = 10
            return i + 15
        return 'foo'
    with pytest.raises(InvalidSubscriptionPlanError):
        test_function(mock_request)

def test_require_plan_does_not_raise_exception_if_plan_valid(rf):
    if False:
        for i in range(10):
            print('nop')
    valid_plan_id = 'plan-id'
    mock_request = MagicMock(spec=Request)
    mock_subscription = MagicMock(plan=valid_plan_id)

    @require_plan([valid_plan_id], lambda v: mock_subscription)
    def test_function(request: Request):
        if False:
            i = 10
            return i + 15
        return 'foo'
    res = test_function(mock_request)
    assert res == 'foo'