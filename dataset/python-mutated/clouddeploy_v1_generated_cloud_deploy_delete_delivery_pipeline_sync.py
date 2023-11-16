from google.cloud import deploy_v1

def sample_delete_delivery_pipeline():
    if False:
        while True:
            i = 10
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.DeleteDeliveryPipelineRequest(name='name_value')
    operation = client.delete_delivery_pipeline(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)