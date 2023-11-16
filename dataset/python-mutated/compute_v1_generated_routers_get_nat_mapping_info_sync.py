from google.cloud import compute_v1

def sample_get_nat_mapping_info():
    if False:
        while True:
            i = 10
    client = compute_v1.RoutersClient()
    request = compute_v1.GetNatMappingInfoRoutersRequest(project='project_value', region='region_value', router='router_value')
    page_result = client.get_nat_mapping_info(request=request)
    for response in page_result:
        print(response)