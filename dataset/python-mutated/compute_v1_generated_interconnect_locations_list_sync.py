from google.cloud import compute_v1

def sample_list():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InterconnectLocationsClient()
    request = compute_v1.ListInterconnectLocationsRequest(project='project_value')
    page_result = client.list(request=request)
    for response in page_result:
        print(response)