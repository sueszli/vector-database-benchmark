from google.cloud import deploy_v1

def sample_update_delivery_pipeline():
    if False:
        for i in range(10):
            print('nop')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.UpdateDeliveryPipelineRequest()
    operation = client.update_delivery_pipeline(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)