from google.cloud import vmwareengine_v1
TIMEOUT = 1200

def create_legacy_network(project_id: str, region: str) -> vmwareengine_v1.VmwareEngineNetwork:
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a new legacy network.\n\n    Args:\n        project_id: name of the project you want to use.\n        region: name of the region you want to use. I.e. "us-central1"\n\n    Returns:\n        The newly created VmwareEngineNetwork object.\n    '
    network = vmwareengine_v1.VmwareEngineNetwork()
    network.description = 'Legacy network created using vmwareengine_v1.VmwareEngineNetwork'
    network.type_ = vmwareengine_v1.VmwareEngineNetwork.Type.LEGACY
    request = vmwareengine_v1.CreateVmwareEngineNetworkRequest()
    request.parent = f'projects/{project_id}/locations/{region}'
    request.vmware_engine_network_id = f'{region}-default'
    request.vmware_engine_network = network
    client = vmwareengine_v1.VmwareEngineClient()
    result = client.create_vmware_engine_network(request, timeout=TIMEOUT).result()
    return result