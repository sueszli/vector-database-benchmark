from google.cloud.devtools import cloudbuild_v1

def sample_delete_worker_pool():
    if False:
        for i in range(10):
            print('nop')
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.DeleteWorkerPoolRequest(name='name_value')
    operation = client.delete_worker_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)