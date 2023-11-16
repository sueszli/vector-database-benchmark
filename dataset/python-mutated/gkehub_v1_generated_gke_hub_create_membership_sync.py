from google.cloud import gkehub_v1

def sample_create_membership():
    if False:
        print('Hello World!')
    client = gkehub_v1.GkeHubClient()
    request = gkehub_v1.CreateMembershipRequest(parent='parent_value', membership_id='membership_id_value')
    operation = client.create_membership(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)