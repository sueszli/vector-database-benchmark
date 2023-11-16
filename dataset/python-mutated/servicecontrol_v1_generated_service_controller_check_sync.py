from google.cloud import servicecontrol_v1

def sample_check():
    if False:
        i = 10
        return i + 15
    client = servicecontrol_v1.ServiceControllerClient()
    request = servicecontrol_v1.CheckRequest()
    response = client.check(request=request)
    print(response)