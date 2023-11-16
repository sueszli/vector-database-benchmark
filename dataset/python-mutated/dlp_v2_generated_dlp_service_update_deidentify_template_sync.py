from google.cloud import dlp_v2

def sample_update_deidentify_template():
    if False:
        return 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.UpdateDeidentifyTemplateRequest(name='name_value')
    response = client.update_deidentify_template(request=request)
    print(response)