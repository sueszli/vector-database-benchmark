from typing import Sequence
import pytest
from dagster import AssetMaterialization, AssetSelection, DagsterInstance, job, op
from dagster._core.definitions.asset_graph import AssetGraph
from dagster._core.definitions.auto_materialize_rule_evaluation import AutoMaterializeAssetEvaluation
from dagster._core.definitions.time_window_partitions import HourlyPartitionsDefinition
from .base_scenario import AssetReconciliationScenario, asset_def
from .scenarios.scenarios import ASSET_RECONCILIATION_SCENARIOS

@pytest.mark.parametrize('respect_materialization_data_versions', [True, False])
@pytest.mark.parametrize('scenario', list(ASSET_RECONCILIATION_SCENARIOS.values()), ids=list(ASSET_RECONCILIATION_SCENARIOS.keys()))
def test_reconciliation(scenario, respect_materialization_data_versions):
    if False:
        return 10
    instance = DagsterInstance.ephemeral()
    (run_requests, _, evaluations) = scenario.do_sensor_scenario(instance, respect_materialization_data_versions=respect_materialization_data_versions)

    def _sorted_evaluations(evaluations: Sequence[AutoMaterializeAssetEvaluation]) -> Sequence[AutoMaterializeAssetEvaluation]:
        if False:
            return 10
        'Allows a stable ordering for comparison.'
        return sorted([evaluation._replace(partition_subsets_by_condition=sorted(evaluation.partition_subsets_by_condition, key=repr))._replace(rule_snapshots=sorted(evaluation.rule_snapshots, key=repr) if evaluation.rule_snapshots else None) for evaluation in evaluations], key=repr)
    if scenario.expected_evaluations is not None:
        asset_graph = AssetGraph.from_assets(scenario.assets)
        assert _sorted_evaluations([evaluation_spec.to_evaluation(asset_graph, instance) for evaluation_spec in scenario.expected_evaluations]) == _sorted_evaluations(evaluations)
    assert len(run_requests) == len(scenario.expected_run_requests), evaluations

    def sort_run_request_key_fn(run_request):
        if False:
            for i in range(10):
                print('nop')
        return (min(run_request.asset_selection), run_request.partition_key)
    sorted_run_requests = sorted(run_requests, key=sort_run_request_key_fn)
    sorted_expected_run_requests = sorted(scenario.expected_run_requests, key=sort_run_request_key_fn)
    for (run_request, expected_run_request) in zip(sorted_run_requests, sorted_expected_run_requests):
        assert set(run_request.asset_selection) == set(expected_run_request.asset_selection)
        assert run_request.partition_key == expected_run_request.partition_key

@pytest.mark.parametrize('scenario', [ASSET_RECONCILIATION_SCENARIOS['freshness_complex_subsettable']])
def test_reconciliation_no_tags(scenario):
    if False:
        while True:
            i = 10
    instance = DagsterInstance.ephemeral()
    (run_requests, _, _) = scenario.do_sensor_scenario(instance)
    assert len(run_requests) == len(scenario.expected_run_requests)

    def sort_run_request_key_fn(run_request):
        if False:
            print('Hello World!')
        return (min(run_request.asset_selection), run_request.partition_key)
    sorted_run_requests = sorted(run_requests, key=sort_run_request_key_fn)
    sorted_expected_run_requests = sorted(scenario.expected_run_requests, key=sort_run_request_key_fn)
    for (run_request, expected_run_request) in zip(sorted_run_requests, sorted_expected_run_requests):
        assert set(run_request.asset_selection) == set(expected_run_request.asset_selection)
        assert run_request.partition_key == expected_run_request.partition_key

def test_bad_partition_key():
    if False:
        for i in range(10):
            print('nop')
    hourly_partitions_def = HourlyPartitionsDefinition('2013-01-05-00:00')
    assets = [asset_def('hourly1', partitions_def=hourly_partitions_def), asset_def('hourly2', ['hourly1'], partitions_def=hourly_partitions_def)]
    instance = DagsterInstance.ephemeral()

    @op
    def materialization_op(context):
        if False:
            for i in range(10):
                print('nop')
        context.log_event(AssetMaterialization('hourly1', partition='bad partition key'))

    @job
    def materialization_job():
        if False:
            return 10
        materialization_op()
    materialization_job.execute_in_process(instance=instance)
    scenario = AssetReconciliationScenario(assets=assets, unevaluated_runs=[], asset_selection=AssetSelection.keys('hourly2'))
    (run_requests, _, _) = scenario.do_sensor_scenario(instance)
    assert len(run_requests) == 0