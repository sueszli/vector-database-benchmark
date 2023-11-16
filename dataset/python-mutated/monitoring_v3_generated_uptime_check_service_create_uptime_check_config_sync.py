from google.cloud import monitoring_v3

def sample_create_uptime_check_config():
    if False:
        print('Hello World!')
    client = monitoring_v3.UptimeCheckServiceClient()
    request = monitoring_v3.CreateUptimeCheckConfigRequest(parent='parent_value')
    response = client.create_uptime_check_config(request=request)
    print(response)