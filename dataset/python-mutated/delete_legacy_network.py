from google.api_core import operation
from google.cloud import vmwareengine_v1

def delete_legacy_network(project_id: str, region: str) -> operation.Operation:
    if False:
        print('Hello World!')
    '\n    Deletes a legacy VMWare Network.\n\n    Args:\n        project_id: name of the project hosting the network.\n        region: region in which the legacy network is located in.\n\n    Returns:\n        An Operation object related to started network deletion operation.\n    '
    client = vmwareengine_v1.VmwareEngineClient()
    return client.delete_vmware_engine_network(name=f'projects/{project_id}/locations/{region}/vmwareEngineNetworks/{region}-default')