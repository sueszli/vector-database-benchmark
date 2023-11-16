from google.cloud import compute_v1

def sample_insert():
    if False:
        while True:
            i = 10
    client = compute_v1.NetworksClient()
    request = compute_v1.InsertNetworkRequest(project='project_value')
    response = client.insert(request=request)
    print(response)