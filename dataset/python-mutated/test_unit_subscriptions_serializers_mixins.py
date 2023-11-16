from unittest.mock import MagicMock
from rest_framework import serializers
from organisations.models import Subscription
from organisations.subscriptions.serializers.mixins import ReadOnlyIfNotValidPlanMixin

def test_read_only_if_not_valid_plan_mixin_sets_read_only_if_plan_not_valid():
    if False:
        while True:
            i = 10
    invalid_plan_id = 'invalid-plan-id'
    mock_view = MagicMock()

    class MySerializer(ReadOnlyIfNotValidPlanMixin, serializers.Serializer):
        invalid_plans = (invalid_plan_id,)
        field_names = ('foo',)
        foo = serializers.CharField()

        def get_subscription(self) -> Subscription:
            if False:
                return 10
            return MagicMock(plan=invalid_plan_id)
    serializer = MySerializer(data={'foo': 'bar'}, context={'view': mock_view})
    serializer.is_valid()
    assert 'foo' not in serializer.validated_data
    assert serializer.fields['foo'].read_only is True

def test_read_only_if_not_valid_plan_mixin_does_not_set_read_only_if_plan_valid():
    if False:
        print('Hello World!')
    valid_plan_id = 'plan-id'
    invalid_plan_id = 'invalid-plan-id'
    mock_view = MagicMock()

    class MySerializer(ReadOnlyIfNotValidPlanMixin, serializers.Serializer):
        invalid_plans = (invalid_plan_id,)
        field_names = ('foo',)
        foo = serializers.CharField()

        def get_subscription(self) -> Subscription:
            if False:
                print('Hello World!')
            return MagicMock(plan=valid_plan_id)
    serializer = MySerializer(data={'foo': 'bar'}, context={'view': mock_view})
    serializer.is_valid()
    assert 'foo' in serializer.validated_data
    assert serializer.fields['foo'].read_only is False