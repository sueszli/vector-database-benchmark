from google.cloud import storage_transfer_v1

def sample_create_agent_pool():
    if False:
        for i in range(10):
            print('nop')
    client = storage_transfer_v1.StorageTransferServiceClient()
    agent_pool = storage_transfer_v1.AgentPool()
    agent_pool.name = 'name_value'
    request = storage_transfer_v1.CreateAgentPoolRequest(project_id='project_id_value', agent_pool=agent_pool, agent_pool_id='agent_pool_id_value')
    response = client.create_agent_pool(request=request)
    print(response)