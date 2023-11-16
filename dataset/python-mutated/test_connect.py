import numpy as np
import skimage.graph.mcp as mcp
from skimage._shared.testing import assert_array_equal
a = np.ones((8, 8), dtype=np.float32)
count = 0

class MCP(mcp.MCP_Connect):

    def _reset(self):
        if False:
            return 10
        'Reset the id map.'
        mcp.MCP_Connect._reset(self)
        self._conn = {}
        self._bestconn = {}

    def create_connection(self, id1, id2, pos1, pos2, cost1, cost2):
        if False:
            for i in range(10):
                print('nop')
        hash = (min(id1, id2), max(id1, id2))
        val = (min(pos1, pos2), max(pos1, pos2))
        cost = min(cost1, cost2)
        self._conn.setdefault(hash, []).append(val)
        curcost = self._bestconn.get(hash, (np.inf,))[0]
        if cost < curcost:
            self._bestconn[hash] = (cost,) + val

def test_connections():
    if False:
        print('Hello World!')
    mcp = MCP(a)
    (costs, traceback) = mcp.find_costs([(1, 1), (7, 7), (1, 7)])
    connections = set(mcp._conn.keys())
    assert (0, 1) in connections
    assert (1, 2) in connections
    assert (0, 2) in connections
    for position_tuples in mcp._conn.values():
        n1 = len(position_tuples)
        n2 = len(set(position_tuples))
        assert n1 == n2
    (cost, pos1, pos2) = mcp._bestconn[0, 1]
    assert (pos1, pos2) == ((3, 3), (4, 4))
    path = mcp.traceback(pos1) + list(reversed(mcp.traceback(pos2)))
    assert_array_equal(path, [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)])
    (cost, pos1, pos2) = mcp._bestconn[1, 2]
    assert (pos1, pos2) == ((3, 7), (4, 7))
    path = mcp.traceback(pos1) + list(reversed(mcp.traceback(pos2)))
    assert_array_equal(path, [(1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7)])
    (cost, pos1, pos2) = mcp._bestconn[0, 2]
    assert (pos1, pos2) == ((1, 3), (1, 4))
    path = mcp.traceback(pos1) + list(reversed(mcp.traceback(pos2)))
    assert_array_equal(path, [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)])