from google.cloud import deploy_v1

def sample_list_delivery_pipelines():
    if False:
        i = 10
        return i + 15
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.ListDeliveryPipelinesRequest(parent='parent_value')
    page_result = client.list_delivery_pipelines(request=request)
    for response in page_result:
        print(response)