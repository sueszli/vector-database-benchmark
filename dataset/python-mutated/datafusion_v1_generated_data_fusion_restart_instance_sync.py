from google.cloud import data_fusion_v1

def sample_restart_instance():
    if False:
        print('Hello World!')
    client = data_fusion_v1.DataFusionClient()
    request = data_fusion_v1.RestartInstanceRequest(name='name_value')
    operation = client.restart_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)