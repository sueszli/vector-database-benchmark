from google.cloud import container_v1

def sample_set_legacy_abac():
    if False:
        i = 10
        return i + 15
    client = container_v1.ClusterManagerClient()
    request = container_v1.SetLegacyAbacRequest(enabled=True)
    response = client.set_legacy_abac(request=request)
    print(response)