from google.api_core import operation
from google.cloud import vmwareengine_v1

def update_network_policy(project_id: str, region: str, internet_access: bool, external_ip: bool) -> operation.Operation:
    if False:
        while True:
            i = 10
    '\n    Updates a network policy in a given network.\n\n    Args:\n        project_id: name of the project you want to use.\n        region: name of the region you want to use. I.e. "us-central1".\n        internet_access: should internet access be allowed.\n        external_ip: should external IP addresses be assigned.\n\n    Returns:\n        An operation object representing the started operation. You can call its .result() method to wait for\n        it to finish.\n    '
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.UpdateNetworkPolicyRequest()
    request.update_mask = 'internetAccess.enabled,externalIp.enabled'
    network_policy = vmwareengine_v1.NetworkPolicy()
    network_policy.name = f'projects/{project_id}/locations/{region}/networkPolicies/{region}-default'
    network_policy.vmware_engine_network = f'projects/{project_id}/locations/{region}/vmwareEngineNetworks/{region}-default'
    network_policy.internet_access.enabled = internet_access
    network_policy.external_ip.enabled = external_ip
    request.network_policy = network_policy
    return client.update_network_policy(request)