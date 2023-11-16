from google.cloud import retail_v2alpha

def sample_list_serving_configs():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2alpha.ServingConfigServiceClient()
    request = retail_v2alpha.ListServingConfigsRequest(parent='parent_value')
    page_result = client.list_serving_configs(request=request)
    for response in page_result:
        print(response)