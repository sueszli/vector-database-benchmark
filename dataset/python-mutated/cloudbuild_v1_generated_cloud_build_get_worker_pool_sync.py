from google.cloud.devtools import cloudbuild_v1

def sample_get_worker_pool():
    if False:
        for i in range(10):
            print('nop')
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.GetWorkerPoolRequest(name='name_value')
    response = client.get_worker_pool(request=request)
    print(response)