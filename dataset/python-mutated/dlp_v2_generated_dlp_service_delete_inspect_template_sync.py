from google.cloud import dlp_v2

def sample_delete_inspect_template():
    if False:
        print('Hello World!')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.DeleteInspectTemplateRequest(name='name_value')
    client.delete_inspect_template(request=request)