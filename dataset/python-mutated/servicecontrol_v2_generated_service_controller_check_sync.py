from google.cloud import servicecontrol_v2

def sample_check():
    if False:
        i = 10
        return i + 15
    client = servicecontrol_v2.ServiceControllerClient()
    request = servicecontrol_v2.CheckRequest()
    response = client.check(request=request)
    print(response)