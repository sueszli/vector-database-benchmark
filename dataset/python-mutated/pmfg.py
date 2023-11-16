"""
This file contains Graph classes, which create NetworkX's Graph objects from matrices.
"""
import heapq
import itertools
from itertools import count
import warnings
import networkx as nx
from matplotlib import pyplot as plt
from mlfinlab.networks.graph import Graph

class PMFG(Graph):
    """
    PMFG class creates and stores the PMFG as an attribute.
    """

    def __init__(self, input_matrix, matrix_type):
        if False:
            for i in range(10):
                print('nop')
        '\n        PMFG class creates the Planar Maximally Filtered Graph and stores it as an attribute.\n\n        :param input_matrix: (pd.Dataframe) Input distance matrix\n        :param matrix_type: (str) Matrix type name (e.g. "distance").\n        '
        pass

    def get_disparity_measure(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Getter method for the dictionary of disparity measure values of cliques.\n\n        :return: (Dict) Returns a dictionary of clique to the disparity measure.\n        '
        pass

    def _calculate_disparity(self):
        if False:
            i = 10
            return i + 15
        '\n        Calculate disparity given in Tumminello M, Aste T, Di Matteo T, Mantegna RN.\n        A tool for filtering information in complex systems.\n        https://arxiv.org/pdf/cond-mat/0501335.pdf\n\n        :return: (Dict) Returns a dictionary of clique to the disparity measure.\n        '
        pass

    def _generate_cliques(self):
        if False:
            while True:
                i = 10
        '\n        Generate cliques from all of the nodes in the PMFG.\n        '
        pass

    def create_pmfg(self, input_matrix):
        if False:
            print('Hello World!')
        '\n        Creates the PMFG matrix from the input matrix of all edges.\n\n        :param input_matrix: (pd.Dataframe) Input matrix with all edges\n        :return: (nx.Graph) Output PMFG matrix\n        '
        pass

    def get_mst_edges(self):
        if False:
            print('Hello World!')
        '\n        Returns the list of MST edges.\n\n        :return: (list) Returns a list of tuples of edges.\n        '
        pass

    def edge_in_mst(self, node1, node2):
        if False:
            return 10
        '\n        Checks whether the edge from node1 to node2 is a part of the MST.\n\n        :param node1: (str) Name of the first node in the edge.\n        :param node2: (str) Name of the second node in the edge.\n        :return: (bool) Returns true if the edge is in the MST. False otherwise.\n        '
        pass

    def get_graph_plot(self):
        if False:
            print('Hello World!')
        '\n        Overrides parent get_graph_plot to plot it in a planar format.\n\n        Returns the graph of the MST with labels.\n        Assumes that the matrix contains stock names as headers.\n\n        :return: (AxesSubplot) Axes with graph plot. Call plt.show() to display this graph.\n        '
        pass