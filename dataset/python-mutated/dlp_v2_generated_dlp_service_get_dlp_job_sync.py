from google.cloud import dlp_v2

def sample_get_dlp_job():
    if False:
        while True:
            i = 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.GetDlpJobRequest(name='name_value')
    response = client.get_dlp_job(request=request)
    print(response)