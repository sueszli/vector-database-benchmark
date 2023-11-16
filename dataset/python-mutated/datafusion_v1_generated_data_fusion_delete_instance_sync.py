from google.cloud import data_fusion_v1

def sample_delete_instance():
    if False:
        print('Hello World!')
    client = data_fusion_v1.DataFusionClient()
    request = data_fusion_v1.DeleteInstanceRequest(name='name_value')
    operation = client.delete_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)