from google.cloud import tpu_v2alpha1

def sample_create_queued_resource():
    if False:
        while True:
            i = 10
    client = tpu_v2alpha1.TpuClient()
    request = tpu_v2alpha1.CreateQueuedResourceRequest(parent='parent_value')
    operation = client.create_queued_resource(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)