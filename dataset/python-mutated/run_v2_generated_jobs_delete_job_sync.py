from google.cloud import run_v2

def sample_delete_job():
    if False:
        for i in range(10):
            print('nop')
    client = run_v2.JobsClient()
    request = run_v2.DeleteJobRequest(name='name_value')
    operation = client.delete_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)