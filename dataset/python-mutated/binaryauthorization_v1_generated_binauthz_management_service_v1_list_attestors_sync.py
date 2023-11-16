from google.cloud import binaryauthorization_v1

def sample_list_attestors():
    if False:
        return 10
    client = binaryauthorization_v1.BinauthzManagementServiceV1Client()
    request = binaryauthorization_v1.ListAttestorsRequest(parent='parent_value')
    page_result = client.list_attestors(request=request)
    for response in page_result:
        print(response)