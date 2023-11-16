"""
This file contains Graph classes, which create NetworkX's Graph objects from matrices.
"""
import networkx as nx
from mlfinlab.networks.graph import Graph

class MST(Graph):
    """
    MST is a subclass of Graph which creates a MST Graph object.
    """

    def __init__(self, matrix, matrix_type, mst_algorithm='kruskal'):
        if False:
            i = 10
            return i + 15
        '\n        Creates a MST Graph object and stores the MST inside graph attribute.\n\n        :param matrix: (pd.Dataframe) Input matrices such as a distance or correlation matrix.\n        :param matrix_type: (str) Name of the matrix type (e.g. "distance" or "correlation").\n        :param mst_algorithm: (str) Valid MST algorithm types include \'kruskal\', \'prim\', or \'boruvka\'.\n            By default, MST algorithm uses Kruskal\'s.\n        '
        pass

    @staticmethod
    def create_mst(matrix, algorithm='kruskal'):
        if False:
            for i in range(10):
                print('nop')
        "\n        This method converts the input matrix into a MST graph.\n\n        :param matrix: (pd.Dataframe) Input matrix.\n        :param algorithm: (str) Valid MST algorithm types include 'kruskal', 'prim', or 'boruvka'.\n            By default, MST algorithm uses Kruskal's.\n        "
        pass