import collections
import logging
LOG = logging.getLogger(__name__)

class BanditMetaAst:
    nodes = collections.OrderedDict()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def add_node(self, node, parent_id, depth):
        if False:
            return 10
        "Add a node to the AST node collection\n\n        :param node: The AST node to add\n        :param parent_id: The ID of the node's parent\n        :param depth: The depth of the node\n        :return: -\n        "
        node_id = hex(id(node))
        LOG.debug('adding node : %s [%s]', node_id, depth)
        self.nodes[node_id] = {'raw': node, 'parent_id': parent_id, 'depth': depth}

    def __str__(self):
        if False:
            return 10
        'Dumps a listing of all of the nodes\n\n        Dumps a listing of all of the nodes for debugging purposes\n        :return: -\n        '
        tmpstr = ''
        for (k, v) in self.nodes.items():
            tmpstr += f'Node: {k}\n'
            tmpstr += f'\t{str(v)}\n'
        tmpstr += f'Length: {len(self.nodes)}\n'
        return tmpstr