from google.cloud import support_v2

def sample_close_case():
    if False:
        for i in range(10):
            print('nop')
    client = support_v2.CaseServiceClient()
    request = support_v2.CloseCaseRequest(name='name_value')
    response = client.close_case(request=request)
    print(response)