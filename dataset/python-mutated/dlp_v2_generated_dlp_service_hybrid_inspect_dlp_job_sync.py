from google.cloud import dlp_v2

def sample_hybrid_inspect_dlp_job():
    if False:
        i = 10
        return i + 15
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.HybridInspectDlpJobRequest(name='name_value')
    response = client.hybrid_inspect_dlp_job(request=request)
    print(response)