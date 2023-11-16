from typing import List, Set, Tuple
from ray.autoscaler._private import constants

class NodeTracker:
    """Map nodes to their corresponding logs.

    We need to be a little careful here. At an given point in time, node_id <->
    ip can be interchangeably used, but the node_id -> ip relation is not
    bijective _across time_ since IP addresses can be reused. Therefore, we
    should treat node_id as the only unique identifier.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.node_mapping = {}
        self.lru_order = []

    def _add_node_mapping(self, node_id: str, value: str):
        if False:
            for i in range(10):
                print('nop')
        if node_id in self.node_mapping:
            return
        assert len(self.lru_order) == len(self.node_mapping)
        if len(self.lru_order) >= constants.AUTOSCALER_MAX_NODES_TRACKED:
            node_id = self.lru_order.pop(0)
            del self.node_mapping[node_id]
        self.node_mapping[node_id] = value
        self.lru_order.append(node_id)

    def track(self, node_id: str, ip: str, node_type: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Begin to track a new node.\n\n        Args:\n            node_id: The node id.\n            ip: The node ip address.\n            node_type: The node type.\n        '
        if node_id not in self.node_mapping:
            self._add_node_mapping(node_id, (ip, node_type))

    def untrack(self, node_id: str):
        if False:
            for i in range(10):
                print('nop')
        "Gracefully stop tracking a node. If a node is intentionally removed from\n        the cluster, we should stop tracking it so we don't mistakenly mark it\n        as failed.\n\n        Args:\n            node_id: The node id which failed.\n        "
        if node_id in self.node_mapping:
            self.lru_order.remove(node_id)
            del self.node_mapping[node_id]

    def get_all_failed_node_info(self, non_failed_ids: Set[str]) -> List[Tuple[str, str]]:
        if False:
            while True:
                i = 10
        'Get the information about all failed nodes. A failed node is any node which\n        we began to track that is not pending or alive (i.e. not failed).\n\n        Args:\n            non_failed_ids: Nodes are failed unless they are in this set.\n\n        Returns:\n            List[Tuple[str, str]]: A list of tuples. Each tuple is the ip\n            address and type of a failed node.\n        '
        failed_nodes = self.node_mapping.keys() - non_failed_ids
        failed_info = []
        for node_id in filter(lambda node_id: node_id in failed_nodes, self.lru_order):
            failed_info.append(self.node_mapping[node_id])
        return failed_info