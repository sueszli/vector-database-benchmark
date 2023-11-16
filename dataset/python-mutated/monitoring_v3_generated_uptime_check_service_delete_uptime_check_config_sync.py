from google.cloud import monitoring_v3

def sample_delete_uptime_check_config():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.UptimeCheckServiceClient()
    request = monitoring_v3.DeleteUptimeCheckConfigRequest(name='name_value')
    client.delete_uptime_check_config(request=request)