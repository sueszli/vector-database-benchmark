from google.cloud import dlp_v2

def sample_cancel_dlp_job():
    if False:
        print('Hello World!')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.CancelDlpJobRequest(name='name_value')
    client.cancel_dlp_job(request=request)