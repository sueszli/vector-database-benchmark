from google.cloud import dlp_v2

def sample_create_dlp_job():
    if False:
        return 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.CreateDlpJobRequest(parent='parent_value')
    response = client.create_dlp_job(request=request)
    print(response)