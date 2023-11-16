from google.cloud import dlp_v2

def sample_finish_dlp_job():
    if False:
        i = 10
        return i + 15
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.FinishDlpJobRequest(name='name_value')
    client.finish_dlp_job(request=request)