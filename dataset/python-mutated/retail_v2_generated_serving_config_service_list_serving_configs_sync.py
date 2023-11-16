from google.cloud import retail_v2

def sample_list_serving_configs():
    if False:
        i = 10
        return i + 15
    client = retail_v2.ServingConfigServiceClient()
    request = retail_v2.ListServingConfigsRequest(parent='parent_value')
    page_result = client.list_serving_configs(request=request)
    for response in page_result:
        print(response)