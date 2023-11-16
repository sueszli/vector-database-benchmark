from google.cloud import assuredworkloads_v1

def sample_get_violation():
    if False:
        while True:
            i = 10
    client = assuredworkloads_v1.AssuredWorkloadsServiceClient()
    request = assuredworkloads_v1.GetViolationRequest(name='name_value')
    response = client.get_violation(request=request)
    print(response)