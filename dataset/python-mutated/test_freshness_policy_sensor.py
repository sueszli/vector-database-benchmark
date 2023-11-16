from collections import defaultdict
import pytest
from dagster import AssetKey, AssetSelection, DagsterInstance, FreshnessPolicy, asset, build_sensor_context, freshness_policy_sensor, materialize, repository
from dagster._core.definitions.freshness_policy_sensor_definition import FreshnessPolicySensorCursor
from dagster._core.errors import DagsterInvalidDefinitionError
from dagster._core.test_utils import instance_for_test

@asset(freshness_policy=FreshnessPolicy(maximum_lag_minutes=0))
def a():
    if False:
        print('Hello World!')
    return 1

@asset
def b(a):
    if False:
        print('Hello World!')
    return a + 1

@asset(freshness_policy=FreshnessPolicy(maximum_lag_minutes=0))
def c(a):
    if False:
        for i in range(10):
            print('nop')
    return a + 2

@asset(freshness_policy=FreshnessPolicy(maximum_lag_minutes=30))
def d(b, c):
    if False:
        return 10
    return b + c

@asset
def e(d):
    if False:
        return 10
    return d + 1

@repository
def my_repo():
    if False:
        return 10
    return [a, b, c, d, e]

def test_repeated_evaluation():
    if False:
        print('Hello World!')
    _minutes_late_by_key = defaultdict(list)

    @freshness_policy_sensor(asset_selection=AssetSelection.all() - AssetSelection.keys('a'))
    def all_sensor(context):
        if False:
            print('Hello World!')
        if len(_minutes_late_by_key[context.asset_key]) == 0:
            assert context.previous_minutes_overdue is None
        else:
            assert context.previous_minutes_overdue == _minutes_late_by_key[context.asset_key][-1]
        _minutes_late_by_key[context.asset_key].append(context.minutes_overdue)
    with instance_for_test() as instance:
        materialize([a, b, c, d, e], instance=instance)
        context = build_sensor_context(instance=instance, cursor=None, repository_name='my_repo', repository_def=my_repo)
        res = all_sensor.evaluate_tick(context)
        new_cursor = res.cursor
        for _ in range(10):
            context = build_sensor_context(instance=instance, cursor=new_cursor, repository_name='my_repo', repository_def=my_repo)
            res = all_sensor.evaluate_tick(context)
            new_cursor = res.cursor
        assert set(_minutes_late_by_key.keys()) == {AssetKey('c'), AssetKey('d')}
        deserialized_cursor = FreshnessPolicySensorCursor.from_json(new_cursor)
        assert deserialized_cursor.minutes_late_by_key[AssetKey('c')] > 0
        assert deserialized_cursor.minutes_late_by_key[AssetKey('d')] == 0

def test_fail_on_return():
    if False:
        return 10

    @freshness_policy_sensor(asset_selection=AssetSelection.all())
    def all_sensor(_context):
        if False:
            print('Hello World!')
        return 1
    context = build_sensor_context(cursor=FreshnessPolicySensorCursor({}).to_json(), repository_name='my_repo', repository_def=my_repo, instance=DagsterInstance.ephemeral())
    with pytest.raises(DagsterInvalidDefinitionError):
        all_sensor.evaluate_tick(context)

def test_fail_on_yield():
    if False:
        return 10

    @freshness_policy_sensor(asset_selection=AssetSelection.all())
    def all_sensor(_context):
        if False:
            return 10
        yield 1
    context = build_sensor_context(cursor=FreshnessPolicySensorCursor({}).to_json(), repository_name='my_repo', repository_def=my_repo, instance=DagsterInstance.ephemeral())
    with pytest.raises(DagsterInvalidDefinitionError):
        all_sensor.evaluate_tick(context)