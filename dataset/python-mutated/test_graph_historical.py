"""Original NetworkX graph tests"""
import networkx
import networkx as nx
from .historical_tests import HistoricalTests

class TestGraphHistorical(HistoricalTests):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        HistoricalTests.setup_class()
        cls.G = nx.Graph