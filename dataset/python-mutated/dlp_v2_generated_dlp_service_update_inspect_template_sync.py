from google.cloud import dlp_v2

def sample_update_inspect_template():
    if False:
        while True:
            i = 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.UpdateInspectTemplateRequest(name='name_value')
    response = client.update_inspect_template(request=request)
    print(response)