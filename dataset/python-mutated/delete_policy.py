from google.api_core import operation
from google.cloud import vmwareengine_v1

def delete_network_policy(project_id: str, region: str) -> operation.Operation:
    if False:
        return 10
    '\n    Delete a Network Policy.\n\n    Args:\n        project_id: name of the project hosting the policy.\n        region: name of the region hosting the policy. I.e. "us-central1"\n\n    Return:\n        Operation object. You can use .result() to wait for it to finish.\n    '
    client = vmwareengine_v1.VmwareEngineClient()
    return client.delete_network_policy(name=f'projects/{project_id}/locations/{region}/networkPolicies/{region}-default')