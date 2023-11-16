from google.cloud import run_v2

def sample_delete_revision():
    if False:
        while True:
            i = 10
    client = run_v2.RevisionsClient()
    request = run_v2.DeleteRevisionRequest(name='name_value')
    operation = client.delete_revision(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)