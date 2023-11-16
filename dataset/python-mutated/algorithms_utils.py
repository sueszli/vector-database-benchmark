import logging
from ludwig.constants import TIED
logger = logging.getLogger(__name__)

def topological_sort(graph_unsorted):
    if False:
        while True:
            i = 10
    'Repeatedly go through all of the nodes in the graph, moving each of the nodes that has all its edges\n    resolved, onto a sequence that forms our sorted graph.\n\n    A node has all of its edges resolved and can be moved once all the nodes its edges point to, have been moved from\n    the unsorted graph onto the sorted one.\n    '
    graph_sorted = []
    graph_unsorted = dict(graph_unsorted)
    while graph_unsorted:
        acyclic = False
        for (node, edges) in list(graph_unsorted.items()):
            if edges is None:
                edges = []
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append((node, edges))
        if not acyclic:
            raise RuntimeError('A cyclic dependency occurred')
    return graph_sorted

def topological_sort_feature_dependencies(features):
    if False:
        i = 10
        return i + 15
    dependencies_graph = {}
    output_features_dict = {}
    for feature in features:
        dependencies = []
        if 'dependencies' in feature:
            dependencies.extend(feature['dependencies'])
        if TIED in feature:
            dependencies.append(feature[TIED])
        dependencies_graph[feature['name']] = dependencies
        output_features_dict[feature['name']] = feature
    return [output_features_dict[node[0]] for node in topological_sort(dependencies_graph)]
if __name__ == '__main__':
    graph_unsorted = [(2, []), (5, [11]), (11, [2, 9, 10]), (7, [11, 8]), (9, []), (10, []), (8, [9]), (3, [10, 8])]
    logger.info(topological_sort(graph_unsorted))
    graph_unsorted = [('macro', ['action', 'contact_type']), ('contact_type', None), ('action', ['contact_type'])]
    logger.info(topological_sort(graph_unsorted))