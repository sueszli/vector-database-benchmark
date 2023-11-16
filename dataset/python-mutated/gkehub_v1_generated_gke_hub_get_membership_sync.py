from google.cloud import gkehub_v1

def sample_get_membership():
    if False:
        return 10
    client = gkehub_v1.GkeHubClient()
    request = gkehub_v1.GetMembershipRequest(name='name_value')
    response = client.get_membership(request=request)
    print(response)