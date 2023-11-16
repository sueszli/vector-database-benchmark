from google.cloud.devtools import cloudbuild_v1

def sample_update_worker_pool():
    if False:
        return 10
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.UpdateWorkerPoolRequest()
    operation = client.update_worker_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)