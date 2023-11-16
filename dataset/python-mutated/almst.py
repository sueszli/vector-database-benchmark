"""
This file contains Graph classes, which create NetworkX's Graph objects from matrices.
"""
import heapq
import itertools
from itertools import count
import networkx as nx
import numpy as np
import pandas as pd
from mlfinlab.networks.graph import Graph

class ALMST(Graph):
    """
    ALMST is a subclass of Graph which creates a ALMST Graph object.
    The ALMST class converts a distance matrix input into a ALMST matrix. This is then used to create a nx.Graph object.
    """

    def __init__(self, matrix, matrix_type, mst_algorithm='kruskal'):
        if False:
            print('Hello World!')
        '\n        Initialises the ALMST and sets the self.graph attribute as the ALMST graph.\n\n        :param matrix: (pd.Dataframe) Input matrices such as a distance or correlation matrix.\n        :param matrix_type: (str) Name of the matrix type (e.g. "distance" or "correlation").\n        :param mst_algorithm: (str) Valid MST algorithm types include \'kruskal\', \'prim\'.\n            By default, MST algorithm uses Kruskal\'s.\n        '
        pass

    @staticmethod
    def create_almst_kruskals(matrix):
        if False:
            i = 10
            return i + 15
        '\n        This method converts the input matrix into a ALMST matrix.\n\n        ! Currently only works with distance input matrix\n\n        :param matrix: (pd.Dataframe) Input matrix.\n        :return: (pd.Dataframe) ALMST matrix with all other edges as 0 values.\n        '
        pass

    @staticmethod
    def _generate_ordered_heap(matrix, clusters):
        if False:
            i = 10
            return i + 15
        '\n        Given the matrix of edges, and the list of clusters, generate a heap ordered by the average distance between the clusters.\n\n        :param matrix: (pd.Dataframe) Input matrix of the distance matrix.\n        :param clusters: (List) A list of clusters, where each list contains a list of nodes within the cluster.\n        :return: (Heap) Returns a heap ordered by the average distance between the clusters.\n        '
        pass

    @staticmethod
    def _calculate_average_distance(matrix, clusters, c_x, c_y):
        if False:
            while True:
                i = 10
        '\n        Given two clusters, calculates the average distance between the two.\n\n        :param matrix: (pd.Dataframe) Input matrix with all edges.\n        :param clusters: (List) List of clusters.\n        :param c_x: (int) Cluster x, where x is the index of the cluster.\n        :param c_y: (int) Cluster y, where y is the index of the cluster.\n        '
        pass

    @staticmethod
    def _get_min_edge(node, cluster, matrix):
        if False:
            while True:
                i = 10
        '\n        Returns the minimum edge tuple given a node and a cluster.\n\n        :param node: (str) String of the node name.\n        :param cluster: (list) List of node names.\n        :param matrix: (pd.DataFrame) A matrix of all edges.\n        :return: (tuple) A tuple of average distance from node to the cluster, and the minimum edge nodes, i and j.\n        '
        pass

    @staticmethod
    def _get_min_edge_clusters(cluster_one, cluster_two, matrix):
        if False:
            return 10
        '\n        Returns a tuple of the minimum edge and the average length for two clusters.\n\n        :param cluster_one: (list) List of node names.\n        :param cluster_two: (list) List of node names.\n        :param matrix: (pd.DataFrame) A matrix of all edges.\n        :return: (tuple) A tuple of average distance between the clusters, and the minimum edge nodes, i and j.\n        '
        pass

    @staticmethod
    def create_almst(matrix):
        if False:
            for i in range(10):
                print('nop')
        "\n        Creates and returns a ALMST given an input matrix using Prim's algorithm.\n\n        :param matrix: (pd.Dataframe) Input distance matrix of all edges.\n        :return: (pd.Dataframe) Returns the ALMST in matrix format.\n        "
        pass

    @staticmethod
    def _add_next_edge(visited, children, matrix, almst_matrix):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds the next edge with the minimum average distance.\n\n        :param visited: (Set) A set of visited nodes.\n        :param children: (Set) A set of children or frontier nodes, to be visited.\n        :param matrix: (pd.Dataframe) Input distance matrix of all edges.\n        :param almst_matrix: (pd.Dataframe) The ALMST matrix.\n\n        :return: (Tuple) Returns the sets visited and children, and the matrix almst_matrix.\n        '
        pass