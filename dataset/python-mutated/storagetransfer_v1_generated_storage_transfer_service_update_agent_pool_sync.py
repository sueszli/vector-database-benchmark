from google.cloud import storage_transfer_v1

def sample_update_agent_pool():
    if False:
        print('Hello World!')
    client = storage_transfer_v1.StorageTransferServiceClient()
    agent_pool = storage_transfer_v1.AgentPool()
    agent_pool.name = 'name_value'
    request = storage_transfer_v1.UpdateAgentPoolRequest(agent_pool=agent_pool)
    response = client.update_agent_pool(request=request)
    print(response)