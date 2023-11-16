from google.cloud import compute_v1

def sample_delete():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InterconnectsClient()
    request = compute_v1.DeleteInterconnectRequest(interconnect='interconnect_value', project='project_value')
    response = client.delete(request=request)
    print(response)