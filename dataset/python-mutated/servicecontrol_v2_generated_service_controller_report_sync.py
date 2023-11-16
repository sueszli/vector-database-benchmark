from google.cloud import servicecontrol_v2

def sample_report():
    if False:
        for i in range(10):
            print('nop')
    client = servicecontrol_v2.ServiceControllerClient()
    request = servicecontrol_v2.ReportRequest()
    response = client.report(request=request)
    print(response)