import random
import networkx as nx

class GraphPositioning:
    """
    This class is for the calculation of the positions of the nodes of
    a given tree and from the perspective of a given central node
    """

    @staticmethod
    def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
        if False:
            print('Hello World!')
        '\n        Taken from: https://bit.ly/2tetWxf\n\n        If the graph is a tree this will return the positions to plot this in a\n        hierarchical layout.\n\n        G: the graph (must be a tree)\n\n        root: the root node of current branch\n        - if the tree is directed and this is not given,\n          the root will be found and used\n        - if the tree is directed and this is given, then the positions\n          will be just for the descendants of this node.\n        - if the tree is undirected and not given, then a random\n          choice will be used.\n\n        width: horizontal space allocated for this branch - avoids overlap\n          with other branches\n\n        vert_gap: gap between levels of hierarchy\n\n        vert_loc: vertical location of root\n\n        xcenter: horizontal location of root\n        '
        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')
        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
            if False:
                return 10
            '\n            see hierarchy_pos docstring for most arguments\n            pos: a dict saying where all nodes go if they have been assigned\n            parent: parent of this branch. - only affects it if non-directed\n            '
            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if children:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
            return pos
        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)