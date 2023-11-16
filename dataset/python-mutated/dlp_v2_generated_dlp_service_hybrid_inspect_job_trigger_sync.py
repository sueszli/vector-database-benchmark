from google.cloud import dlp_v2

def sample_hybrid_inspect_job_trigger():
    if False:
        print('Hello World!')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.HybridInspectJobTriggerRequest(name='name_value')
    response = client.hybrid_inspect_job_trigger(request=request)
    print(response)