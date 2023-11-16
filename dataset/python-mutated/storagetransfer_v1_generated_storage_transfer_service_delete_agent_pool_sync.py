from google.cloud import storage_transfer_v1

def sample_delete_agent_pool():
    if False:
        for i in range(10):
            print('nop')
    client = storage_transfer_v1.StorageTransferServiceClient()
    request = storage_transfer_v1.DeleteAgentPoolRequest(name='name_value')
    client.delete_agent_pool(request=request)