from google.cloud import compute_v1

def sample_insert():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.NodeTemplatesClient()
    request = compute_v1.InsertNodeTemplateRequest(project='project_value', region='region_value')
    response = client.insert(request=request)
    print(response)