from google.cloud.devtools import cloudbuild_v1

def sample_create_worker_pool():
    if False:
        return 10
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.CreateWorkerPoolRequest(parent='parent_value', worker_pool_id='worker_pool_id_value')
    operation = client.create_worker_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)