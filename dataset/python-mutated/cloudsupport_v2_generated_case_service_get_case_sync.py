from google.cloud import support_v2

def sample_get_case():
    if False:
        return 10
    client = support_v2.CaseServiceClient()
    request = support_v2.GetCaseRequest(name='name_value')
    response = client.get_case(request=request)
    print(response)