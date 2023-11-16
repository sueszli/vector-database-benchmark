"""A ``Vertex`` class.

"""
import matplotlib.pyplot as plt

class Vertex:
    """A class for representing a vertex in Graph SLAM.

    Parameters
    ----------
    vertex_id : int
        The vertex's unique ID
    pose : graphslam.pose.se2.PoseSE2
        The pose associated with the vertex
    vertex_index : int, None
        The vertex's index in the graph's ``vertices`` list

    Attributes
    ----------
    id : int
        The vertex's unique ID
    index : int, None
        The vertex's index in the graph's ``vertices`` list
    pose : graphslam.pose.se2.PoseSE2
        The pose associated with the vertex

    """

    def __init__(self, vertex_id, pose, vertex_index=None):
        if False:
            for i in range(10):
                print('nop')
        self.id = vertex_id
        self.pose = pose
        self.index = vertex_index

    def to_g2o(self):
        if False:
            print('Hello World!')
        'Export the vertex to the .g2o format.\n\n        Returns\n        -------\n        str\n            The vertex in .g2o format\n\n        '
        return 'VERTEX_SE2 {} {} {} {}\n'.format(self.id, self.pose[0], self.pose[1], self.pose[2])

    def plot(self, color='r', marker='o', markersize=3):
        if False:
            return 10
        'Plot the vertex.\n\n        Parameters\n        ----------\n        color : str\n            The color that will be used to plot the vertex\n        marker : str\n            The marker that will be used to plot the vertex\n        markersize : int\n            The size of the plotted vertex\n\n        '
        (x, y) = self.pose.position
        plt.plot(x, y, color=color, marker=marker, markersize=markersize)