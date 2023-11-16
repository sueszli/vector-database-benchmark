from google.cloud import dlp_v2

def sample_list_deidentify_templates():
    if False:
        for i in range(10):
            print('nop')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.ListDeidentifyTemplatesRequest(parent='parent_value')
    page_result = client.list_deidentify_templates(request=request)
    for response in page_result:
        print(response)