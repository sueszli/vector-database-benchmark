from google.cloud import dlp_v2

def sample_get_inspect_template():
    if False:
        return 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.GetInspectTemplateRequest(name='name_value')
    response = client.get_inspect_template(request=request)
    print(response)