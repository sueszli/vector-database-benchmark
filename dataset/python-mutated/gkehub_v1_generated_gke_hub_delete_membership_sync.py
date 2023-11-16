from google.cloud import gkehub_v1

def sample_delete_membership():
    if False:
        print('Hello World!')
    client = gkehub_v1.GkeHubClient()
    request = gkehub_v1.DeleteMembershipRequest(name='name_value')
    operation = client.delete_membership(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)