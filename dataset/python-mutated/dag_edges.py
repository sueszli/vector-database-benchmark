from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.models.abstractoperator import AbstractOperator
if TYPE_CHECKING:
    from airflow.models import Operator
    from airflow.models.dag import DAG

def dag_edges(dag: DAG):
    if False:
        return 10
    '\n    Create the list of edges needed to construct the Graph view.\n\n    A special case is made if a TaskGroup is immediately upstream/downstream of another\n    TaskGroup or task. Two proxy nodes named upstream_join_id and downstream_join_id are\n    created for the TaskGroup. Instead of drawing an edge onto every task in the TaskGroup,\n    all edges are directed onto the proxy nodes. This is to cut down the number of edges on\n    the graph.\n\n    For example: A DAG with TaskGroups group1 and group2:\n        group1: task1, task2, task3\n        group2: task4, task5, task6\n\n    group2 is downstream of group1:\n        group1 >> group2\n\n    Edges to add (This avoids having to create edges between every task in group1 and group2):\n        task1 >> downstream_join_id\n        task2 >> downstream_join_id\n        task3 >> downstream_join_id\n        downstream_join_id >> upstream_join_id\n        upstream_join_id >> task4\n        upstream_join_id >> task5\n        upstream_join_id >> task6\n    '
    edges_to_add = set()
    edges_to_skip = set()
    task_group_map = dag.task_group.get_task_group_dict()

    def collect_edges(task_group):
        if False:
            while True:
                i = 10
        'Update edges_to_add and edges_to_skip according to TaskGroups.'
        if isinstance(task_group, AbstractOperator):
            return
        for target_id in task_group.downstream_group_ids:
            target_group = task_group_map[target_id]
            edges_to_add.add((task_group.downstream_join_id, target_group.upstream_join_id))
            for child in task_group.get_leaves():
                edges_to_add.add((child.task_id, task_group.downstream_join_id))
                for target in target_group.get_roots():
                    edges_to_skip.add((child.task_id, target.task_id))
                edges_to_skip.add((child.task_id, target_group.upstream_join_id))
            for child in target_group.get_roots():
                edges_to_add.add((target_group.upstream_join_id, child.task_id))
                edges_to_skip.add((task_group.downstream_join_id, child.task_id))
        for target_id in task_group.downstream_task_ids:
            edges_to_add.add((task_group.downstream_join_id, target_id))
            for child in task_group.get_leaves():
                edges_to_add.add((child.task_id, task_group.downstream_join_id))
                edges_to_skip.add((child.task_id, target_id))
        for source_id in task_group.upstream_task_ids:
            edges_to_add.add((source_id, task_group.upstream_join_id))
            for child in task_group.get_roots():
                edges_to_add.add((task_group.upstream_join_id, child.task_id))
                edges_to_skip.add((source_id, child.task_id))
        for child in task_group.children.values():
            collect_edges(child)
    collect_edges(dag.task_group)
    edges = set()
    setup_teardown_edges = set()
    tasks_to_trace: list[Operator] = dag.roots
    while tasks_to_trace:
        tasks_to_trace_next: list[Operator] = []
        for task in tasks_to_trace:
            for child in task.downstream_list:
                edge = (task.task_id, child.task_id)
                if task.is_setup and child.is_teardown:
                    setup_teardown_edges.add(edge)
                if edge not in edges:
                    edges.add(edge)
                    tasks_to_trace_next.append(child)
        tasks_to_trace = tasks_to_trace_next
    result = []
    for (source_id, target_id) in sorted(edges.union(edges_to_add) - edges_to_skip):
        record = {'source_id': source_id, 'target_id': target_id}
        label = dag.get_edge_info(source_id, target_id).get('label')
        if (source_id, target_id) in setup_teardown_edges:
            record['is_setup_teardown'] = True
        if label:
            record['label'] = label
        result.append(record)
    return result