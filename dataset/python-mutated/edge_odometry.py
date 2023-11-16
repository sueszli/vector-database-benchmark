"""A class for odometry edges.

"""
import numpy as np
import matplotlib.pyplot as plt
EPSILON = 1e-06

class EdgeOdometry:
    """A class for representing odometry edges in Graph SLAM.

    Parameters
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\\Omega_j` associated with the edge
    estimate : graphslam.pose.se2.PoseSE2
        The expected measurement :math:`\\mathbf{z}_j`

    Attributes
    ----------
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices constrained by the edge
    information : np.ndarray
        The information matrix :math:`\\Omega_j` associated with the edge
    estimate : PoseSE2
        The expected measurement :math:`\\mathbf{z}_j`

    """

    def __init__(self, vertex_ids, information, estimate, vertices=None):
        if False:
            for i in range(10):
                print('nop')
        self.vertex_ids = vertex_ids
        self.information = information
        self.estimate = estimate
        self.vertices = vertices

    def calc_error(self):
        if False:
            for i in range(10):
                print('nop')
        'Calculate the error for the edge: :math:`\\mathbf{e}_j \\in \\mathbb{R}^\\bullet`.\n\n        .. math::\n\n           \\mathbf{e}_j = \\mathbf{z}_j - (p_2 \\ominus p_1)\n\n\n        Returns\n        -------\n        np.ndarray\n            The error for the edge\n\n        '
        return (self.estimate - (self.vertices[1].pose - self.vertices[0].pose)).to_compact()

    def calc_chi2(self):
        if False:
            for i in range(10):
                print('nop')
        'Calculate the :math:`\\chi^2` error for the edge.\n\n        .. math::\n\n           \\mathbf{e}_j^T \\Omega_j \\mathbf{e}_j\n\n\n        Returns\n        -------\n        float\n            The :math:`\\chi^2` error for the edge\n\n        '
        err = self.calc_error()
        return np.dot(np.dot(np.transpose(err), self.information), err)

    def calc_chi2_gradient_hessian(self):
        if False:
            return 10
        "Calculate the edge's contributions to the graph's :math:`\\chi^2` error, gradient (:math:`\\mathbf{b}`), and Hessian (:math:`H`).\n\n        Returns\n        -------\n        float\n            The :math:`\\chi^2` error for the edge\n        dict\n            The edge's contribution(s) to the gradient\n        dict\n            The edge's contribution(s) to the Hessian\n\n        "
        chi2 = self.calc_chi2()
        err = self.calc_error()
        jacobians = self.calc_jacobians()
        return (chi2, {v.index: np.dot(np.dot(np.transpose(err), self.information), jacobian) for (v, jacobian) in zip(self.vertices, jacobians)}, {(self.vertices[i].index, self.vertices[j].index): np.dot(np.dot(np.transpose(jacobians[i]), self.information), jacobians[j]) for i in range(len(jacobians)) for j in range(i, len(jacobians))})

    def calc_jacobians(self):
        if False:
            while True:
                i = 10
        "Calculate the Jacobian of the edge's error with respect to each constrained pose.\n\n        .. math::\n\n           \\frac{\\partial}{\\partial \\Delta \\mathbf{x}^k} \\left[ \\mathbf{e}_j(\\mathbf{x}^k \\boxplus \\Delta \\mathbf{x}^k) \\right]\n\n\n        Returns\n        -------\n        list[np.ndarray]\n            The Jacobian matrices for the edge with respect to each constrained pose\n\n        "
        err = self.calc_error()
        dim = len(self.vertices[0].pose.to_compact())
        return [self._calc_jacobian(err, dim, i) for i in range(len(self.vertices))]

    def _calc_jacobian(self, err, dim, vertex_index):
        if False:
            print('Hello World!')
        "Calculate the Jacobian of the edge with respect to the specified vertex's pose.\n\n        Parameters\n        ----------\n        err : np.ndarray\n            The current error for the edge (see :meth:`EdgeOdometry.calc_error`)\n        dim : int\n            The dimensionality of the compact pose representation\n        vertex_index : int\n            The index of the vertex (pose) for which we are computing the Jacobian\n\n        Returns\n        -------\n        np.ndarray\n            The Jacobian of the edge with respect to the specified vertex's pose\n\n        "
        jacobian = np.zeros(err.shape + (dim,))
        p0 = self.vertices[vertex_index].pose.copy()
        for d in range(dim):
            delta_pose = np.zeros(dim)
            delta_pose[d] = EPSILON
            self.vertices[vertex_index].pose += delta_pose
            jacobian[:, d] = (self.calc_error() - err) / EPSILON
            self.vertices[vertex_index].pose = p0.copy()
        return jacobian

    def to_g2o(self):
        if False:
            i = 10
            return i + 15
        'Export the edge to the .g2o format.\n\n        Returns\n        -------\n        str\n            The edge in .g2o format\n\n        '
        return 'EDGE_SE2 {} {} {} {} {} '.format(self.vertex_ids[0], self.vertex_ids[1], self.estimate[0], self.estimate[1], self.estimate[2]) + ' '.join([str(x) for x in self.information[np.triu_indices(3, 0)]]) + '\n'

    def plot(self, color='b'):
        if False:
            return 10
        'Plot the edge.\n\n        Parameters\n        ----------\n        color : str\n            The color that will be used to plot the edge\n\n        '
        xy = np.array([v.pose.position for v in self.vertices])
        plt.plot(xy[:, 0], xy[:, 1], color=color)