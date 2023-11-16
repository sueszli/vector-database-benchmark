from google.cloud import support_v2

def sample_escalate_case():
    if False:
        print('Hello World!')
    client = support_v2.CaseServiceClient()
    request = support_v2.EscalateCaseRequest(name='name_value')
    response = client.escalate_case(request=request)
    print(response)