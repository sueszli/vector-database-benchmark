from google.cloud import dlp_v2

def sample_delete_deidentify_template():
    if False:
        print('Hello World!')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.DeleteDeidentifyTemplateRequest(name='name_value')
    client.delete_deidentify_template(request=request)