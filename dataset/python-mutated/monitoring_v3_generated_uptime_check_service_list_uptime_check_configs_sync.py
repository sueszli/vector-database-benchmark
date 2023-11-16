from google.cloud import monitoring_v3

def sample_list_uptime_check_configs():
    if False:
        while True:
            i = 10
    client = monitoring_v3.UptimeCheckServiceClient()
    request = monitoring_v3.ListUptimeCheckConfigsRequest(parent='parent_value')
    page_result = client.list_uptime_check_configs(request=request)
    for response in page_result:
        print(response)