from google.cloud import servicemanagement_v1

def sample_generate_config_report():
    if False:
        return 10
    client = servicemanagement_v1.ServiceManagerClient()
    request = servicemanagement_v1.GenerateConfigReportRequest()
    response = client.generate_config_report(request=request)
    print(response)