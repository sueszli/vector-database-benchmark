"""VF2Layout pass to find a layout using subgraph isomorphism"""
from enum import Enum
import itertools
import logging
import time
from rustworkx import vf2_mapping
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.layout import vf2_utils
logger = logging.getLogger(__name__)

class VF2LayoutStopReason(Enum):
    """Stop reasons for VF2Layout pass."""
    SOLUTION_FOUND = 'solution found'
    NO_SOLUTION_FOUND = 'nonexistent solution'
    MORE_THAN_2Q = '>2q gates in basis'

class VF2Layout(AnalysisPass):
    """A pass for choosing a Layout of a circuit onto a Coupling graph, as
    a subgraph isomorphism problem, solved by VF2++.

    If a solution is found that means there is a "perfect layout" and that no
    further swap mapping or routing is needed. If a solution is found the layout
    will be set in the property set as ``property_set['layout']``. However, if no
    solution is found, no ``property_set['layout']`` is set. The stopping reason is
    set in ``property_set['VF2Layout_stop_reason']`` in all the cases and will be
    one of the values enumerated in ``VF2LayoutStopReason`` which has the
    following values:

        * ``"solution found"``: If a perfect layout was found.
        * ``"nonexistent solution"``: If no perfect layout was found.
        * ``">2q gates in basis"``: If VF2Layout can't work with basis

    By default, this pass will construct a heuristic scoring map based on
    the error rates in the provided ``target`` (or ``properties`` if ``target``
    is not provided). However, analysis passes can be run prior to this pass
    and set ``vf2_avg_error_map`` in the property set with a :class:`~.ErrorMap`
    instance. If a value is ``NaN`` that is treated as an ideal edge
    For example if an error map is created as::

        from qiskit.transpiler.passes.layout.vf2_utils import ErrorMap

        error_map = ErrorMap(3)
        error_map.add_error((0, 0), 0.0024)
        error_map.add_error((0, 1), 0.01)
        error_map.add_error((1, 1), 0.0032)

    that represents the error map for a 2 qubit target, where the avg 1q error
    rate is ``0.0024`` on qubit 0 and ``0.0032`` on qubit 1. Then the avg 2q
    error rate for gates that operate on (0, 1) is 0.01 and (1, 0) is not
    supported by the target. This will be used for scoring if it's set as the
    ``vf2_avg_error_map`` key in the property set when :class:`~.VF2Layout` is run.
    """

    def __init__(self, coupling_map=None, strict_direction=False, seed=None, call_limit=None, time_limit=None, properties=None, max_trials=None, target=None):
        if False:
            print('Hello World!')
        'Initialize a ``VF2Layout`` pass instance\n\n        Args:\n            coupling_map (CouplingMap): Directed graph representing a coupling map.\n            strict_direction (bool): If True, considers the direction of the coupling map.\n                                     Default is False.\n            seed (int): Sets the seed of the PRNG. -1 Means no node shuffling.\n            call_limit (int): The number of state visits to attempt in each execution of\n                VF2.\n            time_limit (float): The total time limit in seconds to run ``VF2Layout``\n            properties (BackendProperties): The backend properties for the backend. If\n                :meth:`~qiskit.providers.models.BackendProperties.readout_error` is available\n                it is used to score the layout.\n            max_trials (int): The maximum number of trials to run VF2 to find\n                a layout. If this is not specified the number of trials will be limited\n                based on the number of edges in the interaction graph or the coupling graph\n                (whichever is larger) if no other limits are set. If set to a value <= 0 no\n                limit on the number of trials will be set.\n            target (Target): A target representing the backend device to run ``VF2Layout`` on.\n                If specified it will supersede a set value for ``properties`` and\n                ``coupling_map``.\n\n        Raises:\n            TypeError: At runtime, if neither ``coupling_map`` or ``target`` are provided.\n        '
        super().__init__()
        self.target = target
        if target is not None:
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.coupling_map = coupling_map
        self.properties = properties
        self.strict_direction = strict_direction
        self.seed = seed
        self.call_limit = call_limit
        self.time_limit = time_limit
        self.max_trials = max_trials
        self.avg_error_map = None

    def run(self, dag):
        if False:
            return 10
        'run the layout method'
        if self.coupling_map is None:
            raise TranspilerError('coupling_map or target must be specified.')
        self.avg_error_map = self.property_set['vf2_avg_error_map']
        if self.avg_error_map is None:
            self.avg_error_map = vf2_utils.build_average_error_map(self.target, self.properties, self.coupling_map)
        result = vf2_utils.build_interaction_graph(dag, self.strict_direction)
        if result is None:
            self.property_set['VF2Layout_stop_reason'] = VF2LayoutStopReason.MORE_THAN_2Q
            return
        (im_graph, im_graph_node_map, reverse_im_graph_node_map, free_nodes) = result
        scoring_edge_list = vf2_utils.build_edge_list(im_graph)
        scoring_bit_list = vf2_utils.build_bit_list(im_graph, im_graph_node_map)
        (cm_graph, cm_nodes) = vf2_utils.shuffle_coupling_graph(self.coupling_map, self.seed, self.strict_direction)
        if self.target is not None:
            has_operations = set(itertools.chain.from_iterable(self.target.qargs))
            to_remove = set(range(len(cm_nodes))).difference(has_operations)
            if to_remove:
                cm_graph.remove_nodes_from([cm_nodes[i] for i in to_remove])
        if self.max_trials is None and self.call_limit is None and (self.time_limit is None):
            im_graph_edge_count = len(im_graph.edge_list())
            cm_graph_edge_count = len(self.coupling_map.graph.edge_list())
            self.max_trials = max(im_graph_edge_count, cm_graph_edge_count) + 15
        logger.debug('Running VF2 to find mappings')
        mappings = vf2_mapping(cm_graph, im_graph, subgraph=True, id_order=False, induced=False, call_limit=self.call_limit)
        chosen_layout = None
        chosen_layout_score = None
        start_time = time.time()
        trials = 0

        def mapping_to_layout(layout_mapping):
            if False:
                print('Hello World!')
            return Layout({reverse_im_graph_node_map[k]: v for (k, v) in layout_mapping.items()})
        for mapping in mappings:
            trials += 1
            logger.debug('Running trial: %s', trials)
            stop_reason = VF2LayoutStopReason.SOLUTION_FOUND
            layout_mapping = {im_i: cm_nodes[cm_i] for (cm_i, im_i) in mapping.items()}
            if len(cm_graph) == len(im_graph):
                chosen_layout = mapping_to_layout(layout_mapping)
                break
            if self.avg_error_map is None:
                chosen_layout = mapping_to_layout(layout_mapping)
                break
            layout_score = vf2_utils.score_layout(self.avg_error_map, layout_mapping, im_graph_node_map, reverse_im_graph_node_map, im_graph, self.strict_direction, edge_list=scoring_edge_list, bit_list=scoring_bit_list)
            if layout_score == 0.0:
                chosen_layout = mapping_to_layout(layout_mapping)
                break
            logger.debug('Trial %s has score %s', trials, layout_score)
            if chosen_layout is None:
                chosen_layout = mapping_to_layout(layout_mapping)
                chosen_layout_score = layout_score
            elif layout_score < chosen_layout_score:
                layout = mapping_to_layout(layout_mapping)
                logger.debug('Found layout %s has a lower score (%s) than previous best %s (%s)', layout, layout_score, chosen_layout, chosen_layout_score)
                chosen_layout = layout
                chosen_layout_score = layout_score
            if self.max_trials is not None and self.max_trials > 0 and (trials >= self.max_trials):
                logger.debug('Trial %s is >= configured max trials %s', trials, self.max_trials)
                break
            elapsed_time = time.time() - start_time
            if self.time_limit is not None and elapsed_time >= self.time_limit:
                logger.debug('VF2Layout has taken %s which exceeds configured max time: %s', elapsed_time, self.time_limit)
                break
        if chosen_layout is None:
            stop_reason = VF2LayoutStopReason.NO_SOLUTION_FOUND
        else:
            chosen_layout = vf2_utils.map_free_qubits(free_nodes, chosen_layout, cm_graph.num_nodes(), reverse_im_graph_node_map, self.avg_error_map)
            if chosen_layout is None:
                self.property_set['VF2Layout_stop_reason'] = VF2LayoutStopReason.NO_SOLUTION_FOUND
                return
            self.property_set['layout'] = chosen_layout
            for reg in dag.qregs.values():
                self.property_set['layout'].add_register(reg)
        self.property_set['VF2Layout_stop_reason'] = stop_reason