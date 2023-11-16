from google.cloud import dlp_v2

def sample_get_deidentify_template():
    if False:
        print('Hello World!')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.GetDeidentifyTemplateRequest(name='name_value')
    response = client.get_deidentify_template(request=request)
    print(response)