from google.cloud import run_v2

def sample_get_revision():
    if False:
        while True:
            i = 10
    client = run_v2.RevisionsClient()
    request = run_v2.GetRevisionRequest(name='name_value')
    response = client.get_revision(request=request)
    print(response)