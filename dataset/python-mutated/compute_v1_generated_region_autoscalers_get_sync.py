from google.cloud import compute_v1

def sample_get():
    if False:
        while True:
            i = 10
    client = compute_v1.RegionAutoscalersClient()
    request = compute_v1.GetRegionAutoscalerRequest(autoscaler='autoscaler_value', project='project_value', region='region_value')
    response = client.get(request=request)
    print(response)