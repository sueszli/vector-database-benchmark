from typing import Dict, Optional, Set
from ciphey.iface import Config, ParamSpec, registry
from .ausearch import AuSearch, Node

@registry.register
class Perfection(AuSearch):

    @staticmethod
    def getParams() -> Optional[Dict[str, ParamSpec]]:
        if False:
            return 10
        return None

    def findBestNode(self, nodes: Set[Node]) -> Node:
        if False:
            return 10
        return next(iter(nodes))

    def __init__(self, config: Config):
        if False:
            print('Hello World!')
        super().__init__(config)