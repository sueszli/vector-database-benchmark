from google.cloud import compute_v1

def sample_delete():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionAutoscalersClient()
    request = compute_v1.DeleteRegionAutoscalerRequest(autoscaler='autoscaler_value', project='project_value', region='region_value')
    response = client.delete(request=request)
    print(response)