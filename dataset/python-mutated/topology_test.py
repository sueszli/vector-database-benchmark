"""Tests for topology.py."""
from tensorflow.python.platform import test
from tensorflow.python.tpu import topology

class TopologyTest(test.TestCase):

    def testSerialization(self):
        if False:
            print('Hello World!')
        'Tests if the class is able to generate serialized strings.'
        original_topology = topology.Topology(mesh_shape=[1, 1, 1, 2], device_coordinates=[[[0, 0, 0, 0], [0, 0, 0, 1]]])
        serialized_str = original_topology.serialized()
        new_topology = topology.Topology(serialized=serialized_str)
        self.assertAllEqual(original_topology.mesh_shape, new_topology.mesh_shape)
        self.assertAllEqual(original_topology.device_coordinates, new_topology.device_coordinates)
if __name__ == '__main__':
    test.main()