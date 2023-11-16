from ..abstract_node import NodeType
from ..serde.serializable import serializable
from .node import Node

@serializable()
class Enclave(Node):

    def post_init(self) -> None:
        if False:
            print('Hello World!')
        self.node_type = NodeType.ENCLAVE
        super().post_init()