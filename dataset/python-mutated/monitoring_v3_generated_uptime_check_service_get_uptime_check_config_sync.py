from google.cloud import monitoring_v3

def sample_get_uptime_check_config():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.UptimeCheckServiceClient()
    request = monitoring_v3.GetUptimeCheckConfigRequest(name='name_value')
    response = client.get_uptime_check_config(request=request)
    print(response)