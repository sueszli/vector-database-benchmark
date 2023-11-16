import warnings
from typing import Any, Dict, Optional, Union
from elastic_transport import AsyncTransport, Transport
warnings.warn("Importing from the 'elasticsearch.transport' module is deprecated. Instead import from 'elastic_transport'", category=DeprecationWarning, stacklevel=2)

def get_host_info(node_info: Dict[str, Any], host: Dict[str, Union[int, str]]) -> Optional[Dict[str, Union[int, str]]]:
    if False:
        return 10
    "\n    Simple callback that takes the node info from `/_cluster/nodes` and a\n    parsed connection information and return the connection information. If\n    `None` is returned this node will be skipped.\n    Useful for filtering nodes (by proximity for example) or if additional\n    information needs to be provided for the :class:`~elasticsearch.Connection`\n    class. By default master only nodes are filtered out since they shouldn't\n    typically be used for API operations.\n    :arg node_info: node information from `/_cluster/nodes`\n    :arg host: connection information (host, port) extracted from the node info\n    "
    warnings.warn("The 'get_host_info' function is deprecated. Instead use the 'sniff_node_callback' parameter on the client", category=DeprecationWarning, stacklevel=2)
    if node_info.get('roles', []) == ['master']:
        return None
    return host