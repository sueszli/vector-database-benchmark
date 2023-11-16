from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.RegionNotificationEndpointsClient()
    request = compute_v1.GetRegionNotificationEndpointRequest(notification_endpoint='notification_endpoint_value', project='project_value', region='region_value')
    response = client.get(request=request)
    print(response)