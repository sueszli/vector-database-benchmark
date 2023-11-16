from typing import List
from UM.Scene.Iterator import Iterator
from UM.Scene.SceneNode import SceneNode
from functools import cmp_to_key

class OneAtATimeIterator(Iterator.Iterator):
    """Iterator that returns a list of nodes in the order that they need to be printed

    If there is no solution an empty list is returned.
    Take note that the list of nodes can have children (that may or may not contain mesh data)
    """

    def __init__(self, scene_node) -> None:
        if False:
            print('Hello World!')
        super().__init__(scene_node)
        self._hit_map = [[]]
        self._original_node_list = []

    def _fillStack(self) -> None:
        if False:
            return 10
        'Fills the ``_node_stack`` with a list of scene nodes that need to be printed in order. '
        node_list = []
        for node in self._scene_node.getChildren():
            if not issubclass(type(node), SceneNode):
                continue
            if getattr(node, '_outside_buildarea', False):
                continue
            if node.callDecoration('getConvexHull'):
                node_list.append(node)
        if len(node_list) < 2:
            self._node_stack = node_list[:]
            return
        self._original_node_list = node_list[:]
        self._hit_map = [[self._checkHit(i, j) for i in node_list] for j in node_list]
        for a in range(0, len(node_list)):
            for b in range(0, len(node_list)):
                if a != b and self._hit_map[a][b] and self._hit_map[b][a]:
                    return
        sorted(node_list, key=cmp_to_key(self._calculateScore))
        todo_node_list = [_ObjectOrder([], node_list)]
        while len(todo_node_list) > 0:
            current = todo_node_list.pop()
            for node in current.todo:
                if not self._checkHitMultiple(node, current.order) and (not self._checkBlockMultiple(node, current.todo)):
                    new_todo_list = current.todo[:]
                    new_todo_list.remove(node)
                    new_order = current.order[:] + [node]
                    if len(new_todo_list) == 0:
                        self._node_stack = new_order
                        return
                    todo_node_list.append(_ObjectOrder(new_order, new_todo_list))
        self._node_stack = []

    def _checkHitMultiple(self, node: SceneNode, other_nodes: List[SceneNode]) -> bool:
        if False:
            print('Hello World!')
        node_index = self._original_node_list.index(node)
        for other_node in other_nodes:
            other_node_index = self._original_node_list.index(other_node)
            if self._hit_map[node_index][other_node_index]:
                return True
        return False

    def _checkBlockMultiple(self, node: SceneNode, other_nodes: List[SceneNode]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check for a node whether it hits any of the other nodes.\n\n        :param node: The node to check whether it collides with the other nodes.\n        :param other_nodes: The nodes to check for collisions.\n        :return: returns collision between nodes\n        '
        node_index = self._original_node_list.index(node)
        for other_node in other_nodes:
            other_node_index = self._original_node_list.index(other_node)
            if self._hit_map[other_node_index][node_index] and node_index != other_node_index:
                return True
        return False

    def _calculateScore(self, a: SceneNode, b: SceneNode) -> int:
        if False:
            i = 10
            return i + 15
        "Calculate score simply sums the number of other objects it 'blocks'\n\n        :param a: node\n        :param b: node\n        :return: sum of the number of other objects\n        "
        score_a = sum(self._hit_map[self._original_node_list.index(a)])
        score_b = sum(self._hit_map[self._original_node_list.index(b)])
        return score_a - score_b

    def _checkHit(self, a: SceneNode, b: SceneNode) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Checks if a can be printed before b\n\n        :param a: node\n        :param b: node\n        :return: true if a can be printed before b\n        '
        if a == b:
            return False
        a_hit_hull = a.callDecoration('getConvexHullBoundary')
        b_hit_hull = b.callDecoration('getConvexHullHeadFull')
        overlap = a_hit_hull.intersectsPolygon(b_hit_hull)
        if overlap:
            return True
        a_hit_hull = a.callDecoration('getAdhesionArea')
        b_hit_hull = b.callDecoration('getAdhesionArea')
        overlap = a_hit_hull.intersectsPolygon(b_hit_hull)
        if overlap:
            return True
        else:
            return False

class _ObjectOrder:
    """Internal object used to keep track of a possible order in which to print objects."""

    def __init__(self, order: List[SceneNode], todo: List[SceneNode]) -> None:
        if False:
            i = 10
            return i + 15
        'Creates the _ObjectOrder instance.\n\n        :param order: List of indices in which to print objects, ordered by printing order.\n        :param todo: List of indices which are not yet inserted into the order list.\n        '
        self.order = order
        self.todo = todo