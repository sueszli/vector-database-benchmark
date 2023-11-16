from typing import Iterable
from google.cloud import vmwareengine_v1

def list_networks(project_id: str, region: str) -> Iterable[vmwareengine_v1.VmwareEngineNetwork]:
    if False:
        i = 10
        return i + 15
    '\n    Retrieves a list of VMWare Engine networks defined in given region.\n\n    Args:\n        project_id: name of the project you want to use.\n        region: name of the region for which you want to list networks.\n\n    Returns:\n        An iterable collection containing the VMWareEngineNetworks.\n    '
    client = vmwareengine_v1.VmwareEngineClient()
    return client.list_vmware_engine_networks(parent=f'projects/{project_id}/locations/{region}')