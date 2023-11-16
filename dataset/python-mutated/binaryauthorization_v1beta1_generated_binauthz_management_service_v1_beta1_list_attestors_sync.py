from google.cloud import binaryauthorization_v1beta1

def sample_list_attestors():
    if False:
        for i in range(10):
            print('nop')
    client = binaryauthorization_v1beta1.BinauthzManagementServiceV1Beta1Client()
    request = binaryauthorization_v1beta1.ListAttestorsRequest(parent='parent_value')
    page_result = client.list_attestors(request=request)
    for response in page_result:
        print(response)