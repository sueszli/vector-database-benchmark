"""
These methods allows the user to easily deploy graph visualisations given an input file dataframe.
"""
import warnings
import networkx as nx
from mlfinlab.networks.dash_graph import DashGraph, PMFGDash
from mlfinlab.networks.dual_dash_graph import DualDashGraph
from mlfinlab.networks.mst import MST
from mlfinlab.networks.almst import ALMST
from mlfinlab.networks.pmfg import PMFG
from mlfinlab.codependence import get_distance_matrix

def generate_mst_server(log_returns_df, mst_algorithm='kruskal', distance_matrix_type='angular', jupyter=False, colours=None, sizes=None):
    if False:
        print('Hello World!')
    "\n    This method returns a Dash server ready to be run.\n\n    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns\n        with stock names as columns.\n    :param mst_algorithm: (str) A valid MST type such as 'kruskal', 'prim', or 'boruvka'.\n    :param distance_matrix_type: (str) A valid sub type of a distance matrix,\n        namely 'angular', 'abs_angular', 'squared_angular'.\n    :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.\n    :param colours: (Dict) A dictionary of key string for category name and value of a list of indexes\n        corresponding to the node indexes inputted in the initial dataframe.\n    :param sizes: (List) A list of numbers, where the positions correspond to the node indexes inputted\n        in the initial dataframe.\n    :return: (Dash) Returns the Dash app object, which can be run using run_server.\n        Returns a Jupyter Dash object if the parameter jupyter is set to True.\n    "
    pass

def create_input_matrix(log_returns_df, distance_matrix_type):
    if False:
        print('Hello World!')
    "\n    This method returns the distance matrix ready to be inputted into the Graph class.\n\n    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns\n        with stock names as columns.\n    :param distance_matrix_type: (str) A valid sub type of a distance matrix,\n        namely 'angular', 'abs_angular', 'squared_angular'.\n    :return: (pd.Dataframe) A dataframe of a distance matrix.\n    "
    pass

def generate_almst_server(log_returns_df, distance_matrix_type='angular', jupyter=False, colours=None, sizes=None):
    if False:
        while True:
            i = 10
    "\n    This method returns a Dash server ready to be run.\n\n    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns\n        with stock names as columns.\n    :param distance_matrix_type: (str) A valid sub type of a distance matrix,\n        namely 'angular', 'abs_angular', 'squared_angular'.\n    :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.\n    :param colours: (Dict) A dictionary of key string for category name and value of a list of indexes\n        corresponding to the node indexes inputted in the initial dataframe.\n    :param sizes: (List) A list of numbers, where the positions correspond to the node indexes inputted\n        in the initial dataframe.\n    :return: (Dash) Returns the Dash app object, which can be run using run_server.\n        Returns a Jupyter Dash object if the parameter jupyter is set to True.\n    "
    pass

def generate_mst_almst_comparison(log_returns_df, distance_matrix_type='angular', jupyter=False):
    if False:
        print('Hello World!')
    "\n    This method returns a Dash server ready to be run.\n\n    :param log_returns_df: (pd.Dataframe) An input dataframe of log returns\n        with stock names as columns.\n    :param distance_matrix_type: (str) A valid sub type of a distance matrix,\n        namely 'angular', 'abs_angular', 'squared_angular'.\n    :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.\n    :return: (Dash) Returns the Dash app object, which can be run using run_server.\n        Returns a Jupyter Dash object if the parameter jupyter is set to True.\n    "
    pass

def generate_pmfg_server(log_returns_df, input_type='distance', jupyter=False, colours=None, sizes=None):
    if False:
        print('Hello World!')
    '\n      This method returns a PMFGDash server ready to be run.\n\n      :param log_returns_df: (pd.Dataframe) An input dataframe of log returns\n          with stock names as columns.\n      :param input_type: (str) A valid input type correlation or distance. Inputting correlation will add the edges\n          by largest to smallest, instead of smallest to largest.\n      :param jupyter: (bool) True if the user would like to run inside jupyter notebook. False otherwise.\n      :param colours: (Dict) A dictionary of key string for category name and value of a list of indexes\n          corresponding to the node indexes inputted in the initial dataframe.\n      :param sizes: (List) A list of numbers, where the positions correspond to the node indexes inputted\n          in the initial dataframe.\n      :return: (Dash) Returns the Dash app object, which can be run using run_server.\n          Returns a Jupyter Dash object if the parameter jupyter is set to True.\n      '
    pass

def generate_central_peripheral_ranking(nx_graph):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a NetworkX graph, this method generates and returns a ranking of centrality.\n    The input should be a distance based PMFG.\n\n    The ranking combines multiple centrality measures to calculate an overall ranking of how central or peripheral the\n    nodes are.\n    The smaller the ranking, the more peripheral the node is. The larger the ranking, the more central the node is.\n\n    The factors contributing to the ranking include Degree, Eccentricity, Closeness Centrality, Second Order Centrality,\n    Eigen Vector Centrality and Betweenness Centrality. The formula for these measures can be found on the NetworkX\n    documentation (https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html)\n\n    :param nx_graph: (nx.Graph) NetworkX graph object. You can call get_graph() on the MST, ALMST and PMFG to retrieve\n        the nx.Graph.\n    :return: (List) Returns a list of tuples of ranking value to node.\n    '
    pass