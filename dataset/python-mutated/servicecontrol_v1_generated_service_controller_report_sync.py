from google.cloud import servicecontrol_v1

def sample_report():
    if False:
        i = 10
        return i + 15
    client = servicecontrol_v1.ServiceControllerClient()
    request = servicecontrol_v1.ReportRequest()
    response = client.report(request=request)
    print(response)