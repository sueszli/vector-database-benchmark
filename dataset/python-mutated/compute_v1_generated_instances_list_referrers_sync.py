from google.cloud import compute_v1

def sample_list_referrers():
    if False:
        while True:
            i = 10
    client = compute_v1.InstancesClient()
    request = compute_v1.ListReferrersInstancesRequest(instance='instance_value', project='project_value', zone='zone_value')
    page_result = client.list_referrers(request=request)
    for response in page_result:
        print(response)