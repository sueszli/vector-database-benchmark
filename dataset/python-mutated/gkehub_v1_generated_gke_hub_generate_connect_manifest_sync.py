from google.cloud import gkehub_v1

def sample_generate_connect_manifest():
    if False:
        i = 10
        return i + 15
    client = gkehub_v1.GkeHubClient()
    request = gkehub_v1.GenerateConnectManifestRequest(name='name_value')
    response = client.generate_connect_manifest(request=request)
    print(response)