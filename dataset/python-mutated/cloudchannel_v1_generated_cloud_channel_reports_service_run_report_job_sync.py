from google.cloud import channel_v1

def sample_run_report_job():
    if False:
        for i in range(10):
            print('nop')
    client = channel_v1.CloudChannelReportsServiceClient()
    request = channel_v1.RunReportJobRequest(name='name_value')
    operation = client.run_report_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)