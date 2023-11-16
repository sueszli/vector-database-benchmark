from google.cloud import gkehub_v1

def sample_update_membership():
    if False:
        i = 10
        return i + 15
    client = gkehub_v1.GkeHubClient()
    request = gkehub_v1.UpdateMembershipRequest(name='name_value')
    operation = client.update_membership(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)