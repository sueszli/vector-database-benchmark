import numpy as np
import skimage.graph.mcp as mcp
from skimage._shared.testing import assert_array_equal
a = np.ones((8, 8), dtype=np.float32)
a[1::2] *= 2.0

class FlexibleMCP(mcp.MCP_Flexible):
    """Simple MCP subclass that allows the front to travel
    a certain distance from the seed point, and uses a constant
    cost factor that is independent of the cost array.
    """

    def _reset(self):
        if False:
            i = 10
            return i + 15
        mcp.MCP_Flexible._reset(self)
        self._distance = np.zeros((8, 8), dtype=np.float32).ravel()

    def goal_reached(self, index, cumcost):
        if False:
            for i in range(10):
                print('nop')
        if self._distance[index] > 4:
            return 2
        else:
            return 0

    def travel_cost(self, index, new_index, offset_length):
        if False:
            while True:
                i = 10
        return 1.0

    def examine_neighbor(self, index, new_index, offset_length):
        if False:
            print('Hello World!')
        pass

    def update_node(self, index, new_index, offset_length):
        if False:
            i = 10
            return i + 15
        self._distance[new_index] = self._distance[index] + 1

def test_flexible():
    if False:
        i = 10
        return i + 15
    mcp = FlexibleMCP(a)
    (costs, traceback) = mcp.find_costs([(0, 0)])
    assert_array_equal(costs[:4, :4], [[1, 2, 3, 4], [2, 2, 3, 4], [3, 3, 3, 4], [4, 4, 4, 4]])
    assert np.all(costs[-2:, :] == np.inf)
    assert np.all(costs[:, -2:] == np.inf)