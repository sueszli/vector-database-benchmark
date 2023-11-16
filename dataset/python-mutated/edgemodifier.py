from __future__ import annotations
from typing import Sequence
from airflow.models.taskmixin import DAGNode, DependencyMixin
from airflow.utils.task_group import TaskGroup

class EdgeModifier(DependencyMixin):
    """
    Class that represents edge information to be added between two tasks/operators.

    Has shorthand factory functions, like Label("hooray").

    Current implementation supports
        t1 >> Label("Success route") >> t2
        t2 << Label("Success route") << t2

    Note that due to the potential for use in either direction, this waits
    to make the actual connection between both sides until both are declared,
    and will do so progressively if multiple ups/downs are added.

    This and EdgeInfo are related - an EdgeModifier is the Python object you
    use to add information to (potentially multiple) edges, and EdgeInfo
    is the representation of the information for one specific edge.
    """

    def __init__(self, label: str | None=None):
        if False:
            while True:
                i = 10
        self.label = label
        self._upstream: list[DependencyMixin] = []
        self._downstream: list[DependencyMixin] = []

    @property
    def roots(self):
        if False:
            return 10
        return self._downstream

    @property
    def leaves(self):
        if False:
            i = 10
            return i + 15
        return self._upstream

    @staticmethod
    def _make_list(item_or_list: DependencyMixin | Sequence[DependencyMixin]) -> Sequence[DependencyMixin]:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(item_or_list, Sequence):
            return [item_or_list]
        return item_or_list

    def _save_nodes(self, nodes: DependencyMixin | Sequence[DependencyMixin], stream: list[DependencyMixin]):
        if False:
            i = 10
            return i + 15
        from airflow.models.xcom_arg import XComArg
        for node in self._make_list(nodes):
            if isinstance(node, (TaskGroup, XComArg, DAGNode)):
                stream.append(node)
            else:
                raise TypeError(f'Cannot use edge labels with {type(node).__name__}, only tasks, XComArg or TaskGroups')

    def _convert_streams_to_task_groups(self):
        if False:
            while True:
                i = 10
        '\n        Convert a node to a TaskGroup or leave it as a DAGNode.\n\n        Requires both self._upstream and self._downstream.\n\n        To do this, we keep a set of group_ids seen among the streams. If we find that\n        the nodes are from the same TaskGroup, we will leave them as DAGNodes and not\n        convert them to TaskGroups\n        '
        from airflow.models.xcom_arg import XComArg
        group_ids = set()
        for node in [*self._upstream, *self._downstream]:
            if isinstance(node, DAGNode) and node.task_group:
                if node.task_group.is_root:
                    group_ids.add('root')
                else:
                    group_ids.add(node.task_group.group_id)
            elif isinstance(node, TaskGroup):
                group_ids.add(node.group_id)
            elif isinstance(node, XComArg):
                if isinstance(node.operator, DAGNode) and node.operator.task_group:
                    if node.operator.task_group.is_root:
                        group_ids.add('root')
                    else:
                        group_ids.add(node.operator.task_group.group_id)
        if len(group_ids) != 1:
            self._upstream = self._convert_stream_to_task_groups(self._upstream)
            self._downstream = self._convert_stream_to_task_groups(self._downstream)

    def _convert_stream_to_task_groups(self, stream: Sequence[DependencyMixin]) -> Sequence[DependencyMixin]:
        if False:
            for i in range(10):
                print('nop')
        return [node.task_group if isinstance(node, DAGNode) and node.task_group and (not node.task_group.is_root) else node for node in stream]

    def set_upstream(self, other: DependencyMixin | Sequence[DependencyMixin], edge_modifier: EdgeModifier | None=None):
        if False:
            i = 10
            return i + 15
        '\n        Set the given task/list onto the upstream attribute, then attempt to resolve the relationship.\n\n        Providing this also provides << via DependencyMixin.\n        '
        self._save_nodes(other, self._upstream)
        if self._upstream and self._downstream:
            self._convert_streams_to_task_groups()
        for node in self._downstream:
            node.set_upstream(other, edge_modifier=self)

    def set_downstream(self, other: DependencyMixin | Sequence[DependencyMixin], edge_modifier: EdgeModifier | None=None):
        if False:
            return 10
        '\n        Set the given task/list onto the downstream attribute, then attempt to resolve the relationship.\n\n        Providing this also provides >> via DependencyMixin.\n        '
        self._save_nodes(other, self._downstream)
        if self._upstream and self._downstream:
            self._convert_streams_to_task_groups()
        for node in self._upstream:
            node.set_downstream(other, edge_modifier=self)

    def update_relative(self, other: DependencyMixin, upstream: bool=True, edge_modifier: EdgeModifier | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update relative if we\'re not the "main" side of a relationship; still run the same logic.'
        if upstream:
            self.set_upstream(other)
        else:
            self.set_downstream(other)

    def add_edge_info(self, dag, upstream_id: str, downstream_id: str):
        if False:
            while True:
                i = 10
        '\n        Add or update task info on the DAG for this specific pair of tasks.\n\n        Called either from our relationship trigger methods above, or directly\n        by set_upstream/set_downstream in operators.\n        '
        dag.set_edge_info(upstream_id, downstream_id, {'label': self.label})

def Label(label: str):
    if False:
        for i in range(10):
            print('nop')
    'Create an EdgeModifier that sets a human-readable label on the edge.'
    return EdgeModifier(label=label)