from google.cloud import compute_v1

def sample_insert():
    if False:
        while True:
            i = 10
    client = compute_v1.TargetHttpProxiesClient()
    request = compute_v1.InsertTargetHttpProxyRequest(project='project_value')
    response = client.insert(request=request)
    print(response)