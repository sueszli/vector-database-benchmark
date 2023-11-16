from google.api_core import operation
from google.cloud import vmwareengine_v1

def create_network_policy(project_id: str, region: str, ip_range: str, internet_access: bool, external_ip: bool) -> operation.Operation:
    if False:
        while True:
            i = 10
    '\n    Creates a new network policy in a given network.\n\n    Args:\n        project_id: name of the project you want to use.\n        region: name of the region you want to use. I.e. "us-central1"\n        ip_range: the CIDR range to use for internet access and external IP access gateways,\n            in CIDR notation. An RFC 1918 CIDR block with a "/26" suffix is required.\n        internet_access: should internet access be allowed.\n        external_ip: should external IP addresses be assigned.\n\n    Returns:\n        An operation object representing the started operation. You can call its .result() method to wait for\n        it to finish.\n\n    Raises:\n        ValueError if the provided ip_range doesn\'t end with /26.\n    '
    if not ip_range.endswith('/26'):
        raise ValueError("The ip_range needs to be an RFC 1918 CIDR block with a '/26' suffix")
    network_policy = vmwareengine_v1.NetworkPolicy()
    network_policy.vmware_engine_network = f'projects/{project_id}/locations/{region}/vmwareEngineNetworks/{region}-default'
    network_policy.edge_services_cidr = ip_range
    network_policy.internet_access.enabled = internet_access
    network_policy.external_ip.enabled = external_ip
    request = vmwareengine_v1.CreateNetworkPolicyRequest()
    request.network_policy = network_policy
    request.parent = f'projects/{project_id}/locations/{region}'
    request.network_policy_id = f'{region}-default'
    client = vmwareengine_v1.VmwareEngineClient()
    return client.create_network_policy(request)