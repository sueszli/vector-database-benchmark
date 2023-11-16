from google.cloud import iam_v2beta

def sample_list_policies():
    if False:
        i = 10
        return i + 15
    client = iam_v2beta.PoliciesClient()
    request = iam_v2beta.ListPoliciesRequest(parent='parent_value')
    page_result = client.list_policies(request=request)
    for response in page_result:
        print(response)