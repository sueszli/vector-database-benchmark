from google.cloud import compute_v1

def sample_insert():
    if False:
        print('Hello World!')
    client = compute_v1.InterconnectsClient()
    request = compute_v1.InsertInterconnectRequest(project='project_value')
    response = client.insert(request=request)
    print(response)