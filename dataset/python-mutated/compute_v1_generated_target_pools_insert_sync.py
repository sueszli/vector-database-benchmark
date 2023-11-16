from google.cloud import compute_v1

def sample_insert():
    if False:
        while True:
            i = 10
    client = compute_v1.TargetPoolsClient()
    request = compute_v1.InsertTargetPoolRequest(project='project_value', region='region_value')
    response = client.insert(request=request)
    print(response)