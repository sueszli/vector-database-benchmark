from typing import AbstractSet, List, Mapping, Optional, Set, Tuple
from airflow.models.connection import Connection
from airflow.models.dag import DAG
from dagster import AssetKey, AssetsDefinition, GraphDefinition, OutputMapping, TimeWindowPartitionsDefinition
from dagster._core.definitions.graph_definition import create_adjacency_lists
from dagster._utils.schedules import is_valid_cron_schedule
from dagster_airflow.dagster_job_factory import make_dagster_job_from_airflow_dag
from dagster_airflow.utils import DagsterAirflowError, normalized_name

def _build_asset_dependencies(dag: DAG, graph: GraphDefinition, task_ids_by_asset_key: Mapping[AssetKey, AbstractSet[str]], upstream_dependencies_by_asset_key: Mapping[AssetKey, AbstractSet[AssetKey]]) -> Tuple[AbstractSet[OutputMapping], Mapping[str, AssetKey], Mapping[str, Set[AssetKey]]]:
    if False:
        i = 10
        return i + 15
    'Builds the asset dependency graph for a given set of airflow task mappings and a dagster graph.'
    output_mappings = set()
    keys_by_output_name = {}
    internal_asset_deps: dict[str, Set[AssetKey]] = {}
    visited_nodes: dict[str, bool] = {}
    upstream_deps = set()

    def find_upstream_dependency(node_name: str) -> None:
        if False:
            return 10
        'Uses Depth-Firs-Search to find all upstream asset dependencies\n        as described in task_ids_by_asset_key.\n        '
        if visited_nodes[node_name]:
            return
        visited_nodes[node_name] = True
        for output_handle in graph.dependency_structure.all_upstream_outputs_from_node(node_name):
            forward_node = output_handle.node_name
            match = False
            for asset_key in task_ids_by_asset_key:
                if forward_node.replace(f'{normalized_name(dag.dag_id)}__', '') in task_ids_by_asset_key[asset_key]:
                    upstream_deps.add(asset_key)
                    match = True
            if not match:
                find_upstream_dependency(forward_node)
    for asset_key in task_ids_by_asset_key:
        asset_upstream_deps = set()
        for task_id in task_ids_by_asset_key[asset_key]:
            visited_nodes = {s.name: False for s in graph.nodes}
            upstream_deps = set()
            find_upstream_dependency(normalized_name(dag.dag_id, task_id))
            for dep in upstream_deps:
                asset_upstream_deps.add(dep)
            keys_by_output_name[f'result_{normalized_name(dag.dag_id, task_id)}'] = asset_key
            output_mappings.add(OutputMapping(graph_output_name=f'result_{normalized_name(dag.dag_id, task_id)}', mapped_node_name=normalized_name(dag.dag_id, task_id), mapped_node_output_name='airflow_task_complete'))
        for task_id in task_ids_by_asset_key[asset_key]:
            if f'result_{normalized_name(dag.dag_id, task_id)}' in internal_asset_deps:
                internal_asset_deps[f'result_{normalized_name(dag.dag_id, task_id)}'].update(asset_upstream_deps)
            else:
                internal_asset_deps[f'result_{normalized_name(dag.dag_id, task_id)}'] = asset_upstream_deps
    for asset_key in upstream_dependencies_by_asset_key:
        for key in keys_by_output_name:
            if keys_by_output_name[key] == asset_key:
                internal_asset_deps[key].update(upstream_dependencies_by_asset_key[asset_key])
    return (output_mappings, keys_by_output_name, internal_asset_deps)

def load_assets_from_airflow_dag(dag: DAG, task_ids_by_asset_key: Mapping[AssetKey, AbstractSet[str]]={}, upstream_dependencies_by_asset_key: Mapping[AssetKey, AbstractSet[AssetKey]]={}, connections: Optional[List[Connection]]=None) -> List[AssetsDefinition]:
    if False:
        i = 10
        return i + 15
    '[Experimental] Construct Dagster Assets for a given Airflow DAG.\n\n    Args:\n        dag (DAG): The Airflow DAG to compile into a Dagster job\n        task_ids_by_asset_key (Optional[Mapping[AssetKey, AbstractSet[str]]]): A mapping from asset\n            keys to task ids. Used break up the Airflow Dag into multiple SDAs\n        upstream_dependencies_by_asset_key (Optional[Mapping[AssetKey, AbstractSet[AssetKey]]]): A\n            mapping from upstream asset keys to assets provided in task_ids_by_asset_key. Used to\n            declare new upstream SDA depenencies.\n        connections (List[Connection]): List of Airflow Connections to be created in the Airflow DB\n\n    Returns:\n        List[AssetsDefinition]\n    '
    cron_schedule = dag.normalized_schedule_interval
    if cron_schedule is not None and (not is_valid_cron_schedule(str(cron_schedule))):
        raise DagsterAirflowError(f'Invalid cron schedule: {cron_schedule} in DAG {dag.dag_id}')
    job = make_dagster_job_from_airflow_dag(dag, connections=connections)
    graph = job._graph_def
    start_date = dag.start_date if dag.start_date else dag.default_args.get('start_date')
    if start_date is None:
        raise DagsterAirflowError(f'Invalid start_date: {start_date} in DAG {dag.dag_id}')
    (forward_edges, _) = create_adjacency_lists(graph.nodes, graph.dependency_structure)
    leaf_nodes = {node_name.replace(f'{normalized_name(dag.dag_id)}__', '') for (node_name, downstream_nodes) in forward_edges.items() if not downstream_nodes}
    mutated_task_ids_by_asset_key: dict[AssetKey, set[str]] = {}
    if task_ids_by_asset_key is None or task_ids_by_asset_key == {}:
        task_ids_by_asset_key = {AssetKey(dag.dag_id): leaf_nodes}
    else:
        used_nodes: set[str] = set()
        for key in task_ids_by_asset_key:
            used_nodes.update(task_ids_by_asset_key[key])
        mutated_task_ids_by_asset_key[AssetKey(dag.dag_id)] = leaf_nodes - used_nodes
    for key in task_ids_by_asset_key:
        if key not in mutated_task_ids_by_asset_key:
            mutated_task_ids_by_asset_key[key] = set(task_ids_by_asset_key[key])
        else:
            mutated_task_ids_by_asset_key[key].update(task_ids_by_asset_key[key])
    (output_mappings, keys_by_output_name, internal_asset_deps) = _build_asset_dependencies(dag, graph, mutated_task_ids_by_asset_key, upstream_dependencies_by_asset_key)
    new_graph = graph.copy(output_mappings=list(output_mappings))
    asset_def = AssetsDefinition.from_graph(graph_def=new_graph, partitions_def=TimeWindowPartitionsDefinition(cron_schedule=str(cron_schedule), timezone=dag.timezone.name, start=start_date.strftime('%Y-%m-%dT%H:%M:%S'), fmt='%Y-%m-%dT%H:%M:%S') if cron_schedule is not None else None, group_name=dag.dag_id, keys_by_output_name=keys_by_output_name, internal_asset_deps=internal_asset_deps, can_subset=True)
    return [asset_def]