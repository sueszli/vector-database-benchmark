from google.cloud import channel_v1

def sample_fetch_report_results():
    if False:
        for i in range(10):
            print('nop')
    client = channel_v1.CloudChannelReportsServiceClient()
    request = channel_v1.FetchReportResultsRequest(report_job='report_job_value')
    page_result = client.fetch_report_results(request=request)
    for response in page_result:
        print(response)