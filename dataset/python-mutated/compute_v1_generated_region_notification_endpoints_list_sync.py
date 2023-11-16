from google.cloud import compute_v1

def sample_list():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionNotificationEndpointsClient()
    request = compute_v1.ListRegionNotificationEndpointsRequest(project='project_value', region='region_value')
    page_result = client.list(request=request)
    for response in page_result:
        print(response)