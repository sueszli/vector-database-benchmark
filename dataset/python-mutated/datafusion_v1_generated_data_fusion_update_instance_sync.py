from google.cloud import data_fusion_v1

def sample_update_instance():
    if False:
        return 10
    client = data_fusion_v1.DataFusionClient()
    instance = data_fusion_v1.Instance()
    instance.type_ = 'DEVELOPER'
    request = data_fusion_v1.UpdateInstanceRequest(instance=instance)
    operation = client.update_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)