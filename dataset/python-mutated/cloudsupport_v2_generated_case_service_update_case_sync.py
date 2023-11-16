from google.cloud import support_v2

def sample_update_case():
    if False:
        print('Hello World!')
    client = support_v2.CaseServiceClient()
    request = support_v2.UpdateCaseRequest()
    response = client.update_case(request=request)
    print(response)