"""Benchmarks for a certain set of algorithms"""
import networkx as nx
from benchmarks.utils import fetch_drug_interaction_network
from networkx.algorithms import community

class AlgorithmBenchmarks:
    timeout = 120
    nodes = 100
    _graphs = [nx.erdos_renyi_graph(nodes, 0.1), nx.erdos_renyi_graph(nodes, 0.5), nx.erdos_renyi_graph(nodes, 0.9), fetch_drug_interaction_network()]
    params = ['Erdos Renyi (100, 0.1)', 'Erdos Renyi (100, 0.5)', 'Erdos Renyi (100, 0.9)', 'Drug Interaction network']
    param_names = ['graph']

    def setup(self, graph):
        if False:
            print('Hello World!')
        self.graphs_dict = dict(zip(self.params, self._graphs))

    def time_betweenness_centrality(self, graph):
        if False:
            return 10
        _ = nx.betweenness_centrality(self.graphs_dict[graph])

    def time_greedy_modularity_communities(self, graph):
        if False:
            for i in range(10):
                print('nop')
        _ = community.greedy_modularity_communities(self.graphs_dict[graph])

    def time_louvain_communities(self, graph):
        if False:
            return 10
        _ = community.louvain_communities(self.graphs_dict[graph])

    def time_pagerank(self, graph):
        if False:
            for i in range(10):
                print('nop')
        _ = nx.pagerank(self.graphs_dict[graph])

    def time_connected_components(self, graph):
        if False:
            return 10
        _ = list(nx.connected_components(self.graphs_dict[graph]))

    def time_k_core(self, graph):
        if False:
            while True:
                i = 10
        _ = nx.k_core(self.graphs_dict[graph])

    def time_average_clustering(self, graph):
        if False:
            while True:
                i = 10
        _ = nx.average_clustering(self.graphs_dict[graph])