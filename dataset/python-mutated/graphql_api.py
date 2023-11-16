"""Helper functions to work with SAM GraphQLApi resource
"""
from typing import Any, Dict, List, Tuple, Union
SCHEMA_ARTIFACT_PROPERTY = 'SchemaUri'
CODE_ARTIFACT_PROPERTY = 'CodeUri'

def find_all_paths_and_values(property_name: str, graphql_dict: Dict[str, Any]) -> List[Tuple[str, Union[str, Dict]]]:
    if False:
        return 10
    "Find paths to the all properties with property_name and their (properties) values.\n\n    It leverages the knowledge of GraphQLApi structure instead of doing generic search in the graph.\n\n    Parameters\n    ----------\n    property_name\n        Name of the property to look up, for example 'CodeUri'\n    graphql_dict\n        GraphQLApi resource dict\n\n    Returns\n    -------\n        list of tuple (path, value) for all found properties which has property_name\n    "
    resolvers_and_functions = {k: graphql_dict[k] for k in ('Resolvers', 'Functions') if k in graphql_dict}
    stack: List[Tuple[Dict[str, Any], str]] = [(resolvers_and_functions, '')]
    paths_values: List[Tuple[str, Union[str, Dict]]] = []
    while stack:
        (node, path) = stack.pop()
        if isinstance(node, dict):
            for (key, value) in node.items():
                if key == property_name:
                    paths_values.append((f'{path}{key}', value))
                elif isinstance(value, dict):
                    stack.append((value, f'{path}{key}.'))
    return paths_values