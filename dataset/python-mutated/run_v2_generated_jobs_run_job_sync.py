from google.cloud import run_v2

def sample_run_job():
    if False:
        while True:
            i = 10
    client = run_v2.JobsClient()
    request = run_v2.RunJobRequest(name='name_value')
    operation = client.run_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)