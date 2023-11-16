from google.cloud import tpu_v2alpha1

def sample_delete_queued_resource():
    if False:
        print('Hello World!')
    client = tpu_v2alpha1.TpuClient()
    request = tpu_v2alpha1.DeleteQueuedResourceRequest(name='name_value')
    operation = client.delete_queued_resource(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)