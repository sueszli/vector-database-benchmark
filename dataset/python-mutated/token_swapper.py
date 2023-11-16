"""Permutation algorithms for general graphs."""
from __future__ import annotations
import logging
from collections.abc import Mapping
import numpy as np
import rustworkx as rx
from .types import Swap, Permutation
from .util import PermutationCircuit, permutation_circuit
logger = logging.getLogger(__name__)

class ApproximateTokenSwapper:
    """A class for computing approximate solutions to the Token Swapping problem.

    Internally caches the graph and associated datastructures for re-use.
    """

    def __init__(self, graph: rx.PyGraph, seed: int | np.random.Generator | None=None) -> None:
        if False:
            return 10
        'Construct an ApproximateTokenSwapping object.\n\n        Args:\n            graph: Undirected graph represented a coupling map.\n            seed: Seed to use for random trials.\n        '
        self.graph = graph
        self.shortest_paths = rx.graph_distance_matrix(graph)
        if isinstance(seed, np.random.Generator):
            self.seed = seed
        else:
            self.seed = np.random.default_rng(seed)

    def distance(self, vertex0: int, vertex1: int) -> int:
        if False:
            print('Hello World!')
        'Compute the distance between two nodes in `graph`.'
        return self.shortest_paths[vertex0, vertex1]

    def permutation_circuit(self, permutation: Permutation, trials: int=4) -> PermutationCircuit:
        if False:
            for i in range(10):
                print('nop')
        'Perform an approximately optimal Token Swapping algorithm to implement the permutation.\n\n        Args:\n          permutation: The partial mapping to implement in swaps.\n          trials: The number of trials to try to perform the mapping. Minimize over the trials.\n\n        Returns:\n          The circuit to implement the permutation\n        '
        sequential_swaps = self.map(permutation, trials=trials)
        parallel_swaps = [[swap] for swap in sequential_swaps]
        return permutation_circuit(parallel_swaps)

    def map(self, mapping: Mapping[int, int], trials: int=4, parallel_threshold: int=50) -> list[Swap[int]]:
        if False:
            return 10
        'Perform an approximately optimal Token Swapping algorithm to implement the permutation.\n\n        Supports partial mappings (i.e. not-permutations) for graphs with missing tokens.\n\n        Based on the paper: Approximation and Hardness for Token Swapping by Miltzow et al. (2016)\n        ArXiV: https://arxiv.org/abs/1602.05150\n        and generalization based on our own work.\n\n        Args:\n          mapping: The partial mapping to implement in swaps.\n          trials: The number of trials to try to perform the mapping. Minimize over the trials.\n          parallel_threshold: The number of nodes in the graph beyond which the algorithm\n                will use parallel processing\n\n        Returns:\n          The swaps to implement the mapping\n        '
        seed = self.seed.integers(1, 10000)
        return rx.graph_token_swapper(self.graph, mapping, trials, seed, parallel_threshold)