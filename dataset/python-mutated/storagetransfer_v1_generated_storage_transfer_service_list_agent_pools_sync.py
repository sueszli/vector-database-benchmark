from google.cloud import storage_transfer_v1

def sample_list_agent_pools():
    if False:
        print('Hello World!')
    client = storage_transfer_v1.StorageTransferServiceClient()
    request = storage_transfer_v1.ListAgentPoolsRequest(project_id='project_id_value')
    page_result = client.list_agent_pools(request=request)
    for response in page_result:
        print(response)