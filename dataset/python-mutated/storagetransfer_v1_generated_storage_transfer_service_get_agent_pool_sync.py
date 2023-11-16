from google.cloud import storage_transfer_v1

def sample_get_agent_pool():
    if False:
        for i in range(10):
            print('nop')
    client = storage_transfer_v1.StorageTransferServiceClient()
    request = storage_transfer_v1.GetAgentPoolRequest(name='name_value')
    response = client.get_agent_pool(request=request)
    print(response)