from typing import Optional, Tuple
from dagster import AssetKey, DailyPartitionsDefinition, Definitions, HourlyPartitionsDefinition, StaticPartitionsDefinition, asset
from dagster._core.definitions.asset_graph_subset import AssetGraphSubset
from dagster._core.definitions.repository_definition.repository_definition import RepositoryDefinition
from dagster._core.execution.asset_backfill import AssetBackfillData
from dagster._core.instance import DagsterInstance
from dagster._core.test_utils import instance_for_test
from dagster_graphql.client.query import LAUNCH_PARTITION_BACKFILL_MUTATION
from dagster_graphql.test.utils import GqlResult, define_out_of_process_context, execute_dagster_graphql, main_repo_location_name
from dagster_tests.definitions_tests.auto_materialize_tests.scenarios.asset_graphs import root_assets_different_partitions_same_downstream
GET_PARTITION_BACKFILLS_QUERY = '\n  query InstanceBackfillsQuery($cursor: String, $limit: Int) {\n    partitionBackfillsOrError(cursor: $cursor, limit: $limit) {\n      ... on PartitionBackfills {\n        results {\n          id\n          status\n          numPartitions\n          timestamp\n          partitionNames\n          partitionSetName\n          partitionSet {\n            id\n            name\n            mode\n            pipelineName\n            repositoryOrigin {\n              id\n              repositoryName\n              repositoryLocationName\n            }\n          }\n        }\n      }\n    }\n  }\n'
SINGLE_BACKFILL_QUERY = '\n  query SingleBackfillQuery($backfillId: String!) {\n    partitionBackfillOrError(backfillId: $backfillId) {\n      ... on PartitionBackfill {\n        partitionStatuses {\n          results {\n            id\n            partitionName\n            runId\n            runStatus\n          }\n        }\n      }\n    }\n  }\n'
ASSET_BACKFILL_DATA_QUERY = '\n  query BackfillStatusesByAsset($backfillId: String!) {\n    partitionBackfillOrError(backfillId: $backfillId) {\n      ... on PartitionBackfill {\n        assetBackfillData {\n            rootTargetedPartitions {\n                partitionKeys\n                ranges {\n                start\n                end\n                }\n            }\n        }\n        isAssetBackfill\n      }\n    }\n  }\n'
ASSET_BACKFILL_PREVIEW_QUERY = '\nquery assetBackfillPreview($params: AssetBackfillPreviewParams!) {\n  assetBackfillPreview(params: $params) {\n    assetKey {\n      path\n    }\n    partitions {\n      partitionKeys\n      ranges {\n        start\n        end\n      }\n    }\n  }\n}\n'

def get_repo() -> RepositoryDefinition:
    if False:
        for i in range(10):
            print('nop')
    partitions_def = StaticPartitionsDefinition(['a', 'b', 'c'])

    @asset(partitions_def=partitions_def)
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        ...

    @asset(partitions_def=partitions_def)
    def asset2():
        if False:
            while True:
                i = 10
        ...

    @asset()
    def asset3():
        if False:
            print('Hello World!')
        'Non-partitioned asset.'
        ...
    return Definitions(assets=[asset1, asset2, asset3]).get_repository_def()

def get_repo_with_non_partitioned_asset() -> RepositoryDefinition:
    if False:
        while True:
            i = 10
    partitions_def = StaticPartitionsDefinition(['a', 'b', 'c'])

    @asset(partitions_def=partitions_def)
    def asset1():
        if False:
            while True:
                i = 10
        ...

    @asset
    def asset2(asset1):
        if False:
            return 10
        ...
    return Definitions(assets=[asset1, asset2]).get_repository_def()

def get_repo_with_root_assets_different_partitions() -> RepositoryDefinition:
    if False:
        i = 10
        return i + 15
    return Definitions(assets=root_assets_different_partitions_same_downstream).get_repository_def()

def test_launch_asset_backfill_read_only_context():
    if False:
        i = 10
        return i + 15
    repo = get_repo()
    all_asset_keys = repo.asset_graph.materializable_asset_keys
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo', instance, read_only=True) as read_only_context:
            assert read_only_context.read_only
            launch_backfill_result = execute_dagster_graphql(read_only_context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'partitionNames': ['a', 'b'], 'assetSelection': [key.to_graphql_input() for key in all_asset_keys]}})
            assert launch_backfill_result
            assert launch_backfill_result.data
            assert launch_backfill_result.data['launchPartitionBackfill']['__typename'] == 'UnauthorizedError'
        location_name = main_repo_location_name()
        with define_out_of_process_context(__file__, 'get_repo', instance, read_only=True, read_only_locations={location_name: False}) as read_only_context:
            assert read_only_context.read_only
            launch_backfill_result = execute_dagster_graphql(read_only_context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'partitionNames': ['a', 'b'], 'assetSelection': [key.to_graphql_input() for key in all_asset_keys]}})
            assert launch_backfill_result
            assert launch_backfill_result.data
            assert launch_backfill_result.data['launchPartitionBackfill']['__typename'] == 'LaunchBackfillSuccess'
            launch_backfill_result = execute_dagster_graphql(read_only_context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'partitionNames': ['a', 'b'], 'assetSelection': [{'path': ['doesnot', 'exist']}]}})
            assert launch_backfill_result
            assert launch_backfill_result.data
            assert launch_backfill_result.data['launchPartitionBackfill']['__typename'] == 'UnauthorizedError'

def test_launch_asset_backfill_all_partitions():
    if False:
        while True:
            i = 10
    repo = get_repo()
    all_asset_keys = repo.asset_graph.materializable_asset_keys
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            launch_backfill_result = execute_dagster_graphql(context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'assetSelection': [key.to_graphql_input() for key in all_asset_keys], 'allPartitions': True}})
            (backfill_id, asset_backfill_data) = _get_backfill_data(launch_backfill_result, instance, repo)
            target_subset = asset_backfill_data.target_subset
            assert target_subset.asset_keys == all_asset_keys
            all_partition_keys = {'a', 'b', 'c'}
            assert target_subset.get_partitions_subset(AssetKey('asset1')).get_partition_keys() == all_partition_keys
            assert target_subset.get_partitions_subset(AssetKey('asset2')).get_partition_keys() == all_partition_keys

def test_launch_asset_backfill_all_partitions_asset_selection():
    if False:
        print('Hello World!')
    repo = get_repo()
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            launch_backfill_result = execute_dagster_graphql(context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'assetSelection': [AssetKey('asset2').to_graphql_input()], 'allPartitions': True}})
            (backfill_id, asset_backfill_data) = _get_backfill_data(launch_backfill_result, instance, repo)
            target_subset = asset_backfill_data.target_subset
            assert target_subset.asset_keys == {AssetKey('asset2')}
            all_partition_keys = {'a', 'b', 'c'}
            assert target_subset.get_partitions_subset(AssetKey('asset2')).get_partition_keys() == all_partition_keys
            assert not target_subset.get_partitions_subset(AssetKey('asset1')).get_partition_keys()

def test_launch_asset_backfill_partitions_by_asset():
    if False:
        return 10
    repo = get_repo()
    all_asset_keys = repo.asset_graph.materializable_asset_keys
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            launch_backfill_result = execute_dagster_graphql(context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'partitionsByAssets': [{'assetKey': AssetKey('asset1').to_graphql_input(), 'partitions': {'range': {'start': 'b', 'end': 'c'}}}, {'assetKey': AssetKey('asset2').to_graphql_input()}, {'assetKey': AssetKey('asset3').to_graphql_input()}]}})
            (backfill_id, asset_backfill_data) = _get_backfill_data(launch_backfill_result, instance, repo)
            target_subset = asset_backfill_data.target_subset
            assert target_subset.asset_keys == all_asset_keys
            all_partition_keys = {'a', 'b', 'c'}
            assert target_subset.get_partitions_subset(AssetKey('asset1')).get_partition_keys() == {'b', 'c'}
            assert target_subset.get_partitions_subset(AssetKey('asset2')).get_partition_keys() == all_partition_keys
            assert target_subset.non_partitioned_asset_keys == {AssetKey('asset3')}

def test_launch_asset_backfill_all_partitions_root_assets_different_partitions():
    if False:
        while True:
            i = 10
    repo = get_repo_with_root_assets_different_partitions()
    all_asset_keys = repo.asset_graph.materializable_asset_keys
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo_with_root_assets_different_partitions', instance) as context:
            launch_backfill_result = execute_dagster_graphql(context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'assetSelection': [key.to_graphql_input() for key in all_asset_keys], 'allPartitions': True}})
            (backfill_id, asset_backfill_data) = _get_backfill_data(launch_backfill_result, instance, repo)
            target_subset = asset_backfill_data.target_subset
            assert target_subset.get_partitions_subset(AssetKey('root1')).get_partition_keys() == {'a', 'b'}
            assert target_subset.get_partitions_subset(AssetKey('root2')).get_partition_keys() == {'1', '2'}
            assert target_subset.get_partitions_subset(AssetKey('downstream')).get_partition_keys() == {'a', 'b'}

def test_launch_asset_backfill():
    if False:
        for i in range(10):
            print('nop')
    repo = get_repo()
    all_asset_keys = repo.asset_graph.materializable_asset_keys
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            launch_backfill_result = execute_dagster_graphql(context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'partitionNames': ['a', 'b'], 'assetSelection': [key.to_graphql_input() for key in all_asset_keys]}})
            (backfill_id, asset_backfill_data) = _get_backfill_data(launch_backfill_result, instance, repo)
            assert asset_backfill_data.target_subset.asset_keys == all_asset_keys
            get_backfills_result = execute_dagster_graphql(context, GET_PARTITION_BACKFILLS_QUERY, variables={})
            assert not get_backfills_result.errors
            assert get_backfills_result.data
            backfill_results = get_backfills_result.data['partitionBackfillsOrError']['results']
            assert len(backfill_results) == 1
            assert backfill_results[0]['numPartitions'] == 2
            assert backfill_results[0]['id'] == backfill_id
            assert backfill_results[0]['partitionSet'] is None
            assert backfill_results[0]['partitionSetName'] is None
            assert set(backfill_results[0]['partitionNames']) == {'a', 'b'}
            single_backfill_result = execute_dagster_graphql(context, SINGLE_BACKFILL_QUERY, variables={'backfillId': backfill_id})
            assert not single_backfill_result.errors
            assert single_backfill_result.data
            assert single_backfill_result.data['partitionBackfillOrError']['partitionStatuses'] is None

def test_remove_partitions_defs_after_backfill():
    if False:
        for i in range(10):
            print('nop')
    repo = get_repo()
    all_asset_keys = repo.asset_graph.materializable_asset_keys
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            launch_backfill_result = execute_dagster_graphql(context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'partitionNames': ['a', 'b'], 'assetSelection': [key.to_graphql_input() for key in all_asset_keys]}})
            (backfill_id, asset_backfill_data) = _get_backfill_data(launch_backfill_result, instance, repo)
            assert asset_backfill_data.target_subset.asset_keys == all_asset_keys
        with define_out_of_process_context(__file__, 'get_repo_with_non_partitioned_asset', instance) as context:
            get_backfills_result = execute_dagster_graphql(context, GET_PARTITION_BACKFILLS_QUERY, variables={})
            assert not get_backfills_result.errors
            assert get_backfills_result.data
            backfill_results = get_backfills_result.data['partitionBackfillsOrError']['results']
            assert len(backfill_results) == 1
            assert backfill_results[0]['numPartitions'] == 0
            assert backfill_results[0]['id'] == backfill_id
            assert backfill_results[0]['partitionSet'] is None
            assert backfill_results[0]['partitionSetName'] is None
            assert set(backfill_results[0]['partitionNames']) == set()
            single_backfill_result = execute_dagster_graphql(context, SINGLE_BACKFILL_QUERY, variables={'backfillId': backfill_id})
            assert not single_backfill_result.errors
            assert single_backfill_result.data
            assert single_backfill_result.data['partitionBackfillOrError']['partitionStatuses'] is None

def test_launch_asset_backfill_with_non_partitioned_asset():
    if False:
        i = 10
        return i + 15
    repo = get_repo_with_non_partitioned_asset()
    all_asset_keys = repo.asset_graph.materializable_asset_keys
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo_with_non_partitioned_asset', instance) as context:
            launch_backfill_result = execute_dagster_graphql(context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'partitionNames': ['a', 'b'], 'assetSelection': [key.to_graphql_input() for key in all_asset_keys]}})
            (backfill_id, asset_backfill_data) = _get_backfill_data(launch_backfill_result, instance, repo)
            target_subset = asset_backfill_data.target_subset
            assert target_subset.asset_keys == all_asset_keys
            assert target_subset.get_partitions_subset(AssetKey('asset1')).get_partition_keys() == {'a', 'b'}
            assert AssetKey('asset2') in target_subset.non_partitioned_asset_keys
            assert AssetKey('asset2') not in target_subset.partitions_subsets_by_asset_key

def get_daily_hourly_repo() -> RepositoryDefinition:
    if False:
        print('Hello World!')

    @asset(partitions_def=HourlyPartitionsDefinition(start_date='2020-01-01-00:00'))
    def hourly():
        if False:
            while True:
                i = 10
        ...

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2020-01-01'))
    def daily(hourly):
        if False:
            i = 10
            return i + 15
        ...
    return Definitions(assets=[hourly, daily]).get_repository_def()

def test_launch_asset_backfill_with_upstream_anchor_asset():
    if False:
        print('Hello World!')
    repo = get_daily_hourly_repo()
    all_asset_keys = repo.asset_graph.materializable_asset_keys
    hourly_partitions = ['2020-01-02-23:00', '2020-01-02-22:00', '2020-01-03-00:00']
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_daily_hourly_repo', instance) as context:
            launch_backfill_result = execute_dagster_graphql(context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'partitionNames': hourly_partitions, 'assetSelection': [key.to_graphql_input() for key in all_asset_keys]}})
            (backfill_id, asset_backfill_data) = _get_backfill_data(launch_backfill_result, instance, repo)
            target_subset = asset_backfill_data.target_subset
            asset_graph = target_subset.asset_graph
            assert target_subset == AssetGraphSubset(target_subset.asset_graph, partitions_subsets_by_asset_key={AssetKey('hourly'): asset_graph.get_partitions_def(AssetKey('hourly')).subset_with_partition_keys(hourly_partitions), AssetKey('daily'): asset_graph.get_partitions_def(AssetKey('daily')).subset_with_partition_keys(['2020-01-02', '2020-01-03'])})
            get_backfills_result = execute_dagster_graphql(context, GET_PARTITION_BACKFILLS_QUERY, variables={})
            assert not get_backfills_result.errors
            assert get_backfills_result.data
            backfill_results = get_backfills_result.data['partitionBackfillsOrError']['results']
            assert len(backfill_results) == 1
            assert backfill_results[0]['numPartitions'] is None
            assert backfill_results[0]['id'] == backfill_id
            assert backfill_results[0]['partitionSet'] is None
            assert backfill_results[0]['partitionSetName'] is None
            assert backfill_results[0]['partitionNames'] is None

def get_daily_two_hourly_repo() -> RepositoryDefinition:
    if False:
        i = 10
        return i + 15

    @asset(partitions_def=HourlyPartitionsDefinition(start_date='2020-01-01-00:00'))
    def hourly1():
        if False:
            return 10
        ...

    @asset(partitions_def=HourlyPartitionsDefinition(start_date='2020-01-01-00:00'))
    def hourly2():
        if False:
            i = 10
            return i + 15
        ...

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2020-01-01'))
    def daily(hourly1, hourly2):
        if False:
            i = 10
            return i + 15
        ...
    return Definitions(assets=[hourly1, hourly2, daily]).get_repository_def()

def test_launch_asset_backfill_with_two_anchor_assets():
    if False:
        print('Hello World!')
    repo = get_daily_two_hourly_repo()
    all_asset_keys = repo.asset_graph.materializable_asset_keys
    hourly_partitions = ['2020-01-02-23:00', '2020-01-02-22:00', '2020-01-03-00:00']
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_daily_two_hourly_repo', instance) as context:
            launch_backfill_result = execute_dagster_graphql(context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'partitionNames': hourly_partitions, 'assetSelection': [key.to_graphql_input() for key in all_asset_keys]}})
            (backfill_id, asset_backfill_data) = _get_backfill_data(launch_backfill_result, instance, repo)
            target_subset = asset_backfill_data.target_subset
            asset_graph = target_subset.asset_graph
            assert target_subset == AssetGraphSubset(target_subset.asset_graph, partitions_subsets_by_asset_key={AssetKey('hourly1'): asset_graph.get_partitions_def(AssetKey('hourly1')).subset_with_partition_keys(hourly_partitions), AssetKey('hourly2'): asset_graph.get_partitions_def(AssetKey('hourly2')).subset_with_partition_keys(hourly_partitions), AssetKey('daily'): asset_graph.get_partitions_def(AssetKey('daily')).subset_with_partition_keys(['2020-01-02', '2020-01-03'])})

def get_daily_hourly_non_partitioned_repo() -> RepositoryDefinition:
    if False:
        for i in range(10):
            print('nop')

    @asset(partitions_def=HourlyPartitionsDefinition(start_date='2020-01-01-00:00'))
    def hourly():
        if False:
            return 10
        ...

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2020-01-01'))
    def daily(hourly):
        if False:
            i = 10
            return i + 15
        ...

    @asset
    def non_partitioned(hourly):
        if False:
            i = 10
            return i + 15
        ...
    return Definitions(assets=[hourly, daily, non_partitioned]).get_repository_def()

def test_launch_asset_backfill_with_upstream_anchor_asset_and_non_partitioned_asset():
    if False:
        i = 10
        return i + 15
    repo = get_daily_hourly_non_partitioned_repo()
    all_asset_keys = repo.asset_graph.materializable_asset_keys
    hourly_partitions = ['2020-01-02-23:00', '2020-01-02-22:00', '2020-01-03-00:00']
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_daily_hourly_non_partitioned_repo', instance) as context:
            launch_backfill_result = execute_dagster_graphql(context, LAUNCH_PARTITION_BACKFILL_MUTATION, variables={'backfillParams': {'partitionNames': hourly_partitions, 'assetSelection': [key.to_graphql_input() for key in all_asset_keys]}})
            (backfill_id, asset_backfill_data) = _get_backfill_data(launch_backfill_result, instance, repo)
            target_subset = asset_backfill_data.target_subset
            asset_graph = target_subset.asset_graph
            assert target_subset == AssetGraphSubset(target_subset.asset_graph, non_partitioned_asset_keys={AssetKey('non_partitioned')}, partitions_subsets_by_asset_key={AssetKey('hourly'): asset_graph.get_partitions_def(AssetKey('hourly')).empty_subset().with_partition_keys(hourly_partitions), AssetKey('daily'): asset_graph.get_partitions_def(AssetKey('daily')).empty_subset().with_partition_keys(['2020-01-02', '2020-01-03'])})
            asset_backfill_data_result = execute_dagster_graphql(context, ASSET_BACKFILL_DATA_QUERY, variables={'backfillId': backfill_id})
            assert asset_backfill_data_result.data
            assert asset_backfill_data_result.data['partitionBackfillOrError']['isAssetBackfill'] is True
            targeted_ranges = asset_backfill_data_result.data['partitionBackfillOrError']['assetBackfillData']['rootTargetedPartitions']['ranges']
            assert len(targeted_ranges) == 1
            assert targeted_ranges[0]['start'] == '2020-01-02-22:00'
            assert targeted_ranges[0]['end'] == '2020-01-03-00:00'

def test_asset_backfill_preview_time_partitioned():
    if False:
        while True:
            i = 10
    repo = get_daily_hourly_non_partitioned_repo()
    all_asset_keys = repo.asset_graph.materializable_asset_keys
    hourly_partitions = ['2020-01-02-23:00', '2020-01-02-22:00', '2020-01-03-00:00']
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_daily_hourly_non_partitioned_repo', instance) as context:
            backfill_preview_result = execute_dagster_graphql(context, ASSET_BACKFILL_PREVIEW_QUERY, variables={'params': {'assetSelection': [key.to_graphql_input() for key in all_asset_keys], 'partitionNames': hourly_partitions}})
            target_asset_partitions = backfill_preview_result.data['assetBackfillPreview']
            assert len(target_asset_partitions) == 3
            assert target_asset_partitions[0]['assetKey'] == {'path': ['hourly']}
            assert target_asset_partitions[0]['partitions']['ranges'] == [{'start': '2020-01-02-22:00', 'end': '2020-01-03-00:00'}]
            assert target_asset_partitions[0]['partitions']['partitionKeys'] is None
            assert target_asset_partitions[1]['assetKey'] == {'path': ['daily']}
            assert target_asset_partitions[1]['partitions']['ranges'] == [{'start': '2020-01-02', 'end': '2020-01-03'}]
            assert target_asset_partitions[1]['partitions']['partitionKeys'] is None
            assert target_asset_partitions[2]['assetKey'] == {'path': ['non_partitioned']}
            assert target_asset_partitions[2]['partitions'] is None

def test_asset_backfill_preview_static_partitioned():
    if False:
        for i in range(10):
            print('nop')
    repo = get_repo()
    all_asset_keys = repo.asset_graph.materializable_asset_keys
    partition_keys = ['a', 'b']
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            backfill_preview_result = execute_dagster_graphql(context, ASSET_BACKFILL_PREVIEW_QUERY, variables={'params': {'partitionNames': partition_keys, 'assetSelection': [key.to_graphql_input() for key in all_asset_keys]}})
            target_asset_partitions = backfill_preview_result.data['assetBackfillPreview']
            assert len(target_asset_partitions) == 3
            assert target_asset_partitions[0]['assetKey'] == {'path': ['asset1']}
            assert target_asset_partitions[0]['partitions']['ranges'] is None
            assert set(target_asset_partitions[0]['partitions']['partitionKeys']) == set(partition_keys)
            assert target_asset_partitions[1]['assetKey'] == {'path': ['asset2']}
            assert target_asset_partitions[1]['partitions']['ranges'] is None
            assert set(target_asset_partitions[1]['partitions']['partitionKeys']) == set(partition_keys)
            assert target_asset_partitions[2]['assetKey'] == {'path': ['asset3']}
            assert target_asset_partitions[2]['partitions'] is None

def _get_backfill_data(launch_backfill_result: GqlResult, instance: DagsterInstance, repo) -> Tuple[str, AssetBackfillData]:
    if False:
        while True:
            i = 10
    assert launch_backfill_result
    assert launch_backfill_result.data
    assert 'backfillId' in launch_backfill_result.data['launchPartitionBackfill'], _get_error_message(launch_backfill_result)
    backfill_id = launch_backfill_result.data['launchPartitionBackfill']['backfillId']
    backfills = instance.get_backfills()
    assert len(backfills) == 1
    backfill = backfills[0]
    assert backfill.backfill_id == backfill_id
    assert backfill.serialized_asset_backfill_data
    return (backfill_id, AssetBackfillData.from_serialized(backfill.serialized_asset_backfill_data, repo.asset_graph, backfill.backfill_timestamp))

def _get_error_message(launch_backfill_result: GqlResult) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    return ''.join(launch_backfill_result.data['launchPartitionBackfill']['stack']) + launch_backfill_result.data['launchPartitionBackfill']['message'] if 'message' in launch_backfill_result.data['launchPartitionBackfill'] else None