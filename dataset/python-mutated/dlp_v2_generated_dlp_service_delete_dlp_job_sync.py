from google.cloud import dlp_v2

def sample_delete_dlp_job():
    if False:
        for i in range(10):
            print('nop')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.DeleteDlpJobRequest(name='name_value')
    client.delete_dlp_job(request=request)