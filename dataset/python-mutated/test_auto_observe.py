import logging
from dagster import AssetKey, DagsterInstance, observable_source_asset
from dagster._core.definitions.asset_daemon_context import AssetDaemonContext, get_auto_observe_run_requests
from dagster._core.definitions.asset_daemon_cursor import AssetDaemonCursor
from dagster._core.definitions.asset_graph import AssetGraph
from pytest import fixture

def test_single_observable_source_asset_no_auto_observe():
    if False:
        i = 10
        return i + 15

    @observable_source_asset
    def asset1():
        if False:
            while True:
                i = 10
        ...
    asset_graph = AssetGraph.from_assets([asset1])
    assert len(get_auto_observe_run_requests(asset_graph=asset_graph, current_timestamp=1000, last_observe_request_timestamp_by_asset_key={}, run_tags={})) == 0
    assert len(get_auto_observe_run_requests(asset_graph=asset_graph, current_timestamp=1000, last_observe_request_timestamp_by_asset_key={AssetKey('asset1'): 1}, run_tags={})) == 0

@fixture
def single_auto_observe_source_asset_graph():
    if False:
        i = 10
        return i + 15

    @observable_source_asset(auto_observe_interval_minutes=30)
    def asset1():
        if False:
            i = 10
            return i + 15
        ...
    asset_graph = AssetGraph.from_assets([asset1])
    return asset_graph

def test_single_observable_source_asset_no_prior_observe_requests(single_auto_observe_source_asset_graph):
    if False:
        return 10
    run_requests = get_auto_observe_run_requests(asset_graph=single_auto_observe_source_asset_graph, current_timestamp=1000, last_observe_request_timestamp_by_asset_key={}, run_tags={})
    assert len(run_requests) == 1
    run_request = run_requests[0]
    assert run_request.asset_selection == [AssetKey('asset1')]

def test_single_observable_source_asset_prior_observe_requests(single_auto_observe_source_asset_graph):
    if False:
        return 10
    last_timestamp = 1000
    run_requests = get_auto_observe_run_requests(asset_graph=single_auto_observe_source_asset_graph, current_timestamp=last_timestamp + 30 * 60 + 5, last_observe_request_timestamp_by_asset_key={AssetKey('asset1'): last_timestamp}, run_tags={})
    assert len(run_requests) == 1
    run_request = run_requests[0]
    assert run_request.asset_selection == [AssetKey('asset1')]

def test_single_observable_source_asset_prior_recent_observe_requests(single_auto_observe_source_asset_graph):
    if False:
        return 10
    last_timestamp = 1000
    run_requests = get_auto_observe_run_requests(asset_graph=single_auto_observe_source_asset_graph, current_timestamp=last_timestamp + 30 * 60 - 5, last_observe_request_timestamp_by_asset_key={AssetKey('asset1'): last_timestamp}, run_tags={})
    assert len(run_requests) == 0

def test_reconcile():
    if False:
        return 10

    @observable_source_asset(auto_observe_interval_minutes=30)
    def asset1():
        if False:
            return 10
        ...
    asset_graph = AssetGraph.from_assets([asset1])
    instance = DagsterInstance.ephemeral()
    (run_requests, cursor, _) = AssetDaemonContext(evaluation_id=1, auto_observe=True, asset_graph=asset_graph, target_asset_keys=set(), instance=instance, cursor=AssetDaemonCursor.empty(), materialize_run_tags=None, observe_run_tags={'tag1': 'tag_value'}, respect_materialization_data_versions=False, logger=logging.getLogger('dagster.amp')).evaluate()
    assert len(run_requests) == 1
    assert run_requests[0].tags.get('tag1') == 'tag_value'
    assert run_requests[0].asset_selection == [AssetKey(['asset1'])]
    assert cursor.last_observe_request_timestamp_by_asset_key[AssetKey(['asset1'])] > 0