from google.cloud import monitoring_v3

def sample_list_uptime_check_ips():
    if False:
        return 10
    client = monitoring_v3.UptimeCheckServiceClient()
    request = monitoring_v3.ListUptimeCheckIpsRequest()
    page_result = client.list_uptime_check_ips(request=request)
    for response in page_result:
        print(response)