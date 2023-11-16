from google.cloud import dlp_v2

def sample_list_inspect_templates():
    if False:
        return 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.ListInspectTemplatesRequest(parent='parent_value')
    page_result = client.list_inspect_templates(request=request)
    for response in page_result:
        print(response)