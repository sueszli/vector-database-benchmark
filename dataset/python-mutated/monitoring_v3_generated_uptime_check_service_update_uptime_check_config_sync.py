from google.cloud import monitoring_v3

def sample_update_uptime_check_config():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.UptimeCheckServiceClient()
    request = monitoring_v3.UpdateUptimeCheckConfigRequest()
    response = client.update_uptime_check_config(request=request)
    print(response)