from google.cloud import support_v2

def sample_create_case():
    if False:
        print('Hello World!')
    client = support_v2.CaseServiceClient()
    request = support_v2.CreateCaseRequest(parent='parent_value')
    response = client.create_case(request=request)
    print(response)