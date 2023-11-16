from google.cloud import servicecontrol_v1

def sample_allocate_quota():
    if False:
        for i in range(10):
            print('nop')
    client = servicecontrol_v1.QuotaControllerClient()
    request = servicecontrol_v1.AllocateQuotaRequest()
    response = client.allocate_quota(request=request)
    print(response)