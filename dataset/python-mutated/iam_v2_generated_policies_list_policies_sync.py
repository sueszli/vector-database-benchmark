from google.cloud import iam_v2

def sample_list_policies():
    if False:
        return 10
    client = iam_v2.PoliciesClient()
    request = iam_v2.ListPoliciesRequest(parent='parent_value')
    page_result = client.list_policies(request=request)
    for response in page_result:
        print(response)