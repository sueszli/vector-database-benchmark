from google.cloud import dlp_v2

def sample_create_inspect_template():
    if False:
        print('Hello World!')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.CreateInspectTemplateRequest(parent='parent_value')
    response = client.create_inspect_template(request=request)
    print(response)