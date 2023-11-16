from google.cloud import retail_v2beta

def sample_list_serving_configs():
    if False:
        i = 10
        return i + 15
    client = retail_v2beta.ServingConfigServiceClient()
    request = retail_v2beta.ListServingConfigsRequest(parent='parent_value')
    page_result = client.list_serving_configs(request=request)
    for response in page_result:
        print(response)