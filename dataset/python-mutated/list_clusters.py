from typing import Iterable
from google.cloud import vmwareengine_v1

def list_clusters(project_id: str, zone: str, private_cloud_name: str) -> Iterable[vmwareengine_v1.Cluster]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Retrieves a list of clusters in private cloud.\n\n    Args:\n        project_id: name of the project hosting the private cloud.\n        zone: zone in which the private cloud is located.\n        private_cloud_name: name of the cloud of which you want to list cluster.\n\n    Returns:\n        An iterable collection of Cluster objects.\n    '
    client = vmwareengine_v1.VmwareEngineClient()
    return client.list_clusters(parent=f'projects/{project_id}/locations/{zone}/privateClouds/{private_cloud_name}')