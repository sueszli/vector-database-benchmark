from google.cloud import dialogflowcx_v3beta1

def sample_get_changelog():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3beta1.ChangelogsClient()
    request = dialogflowcx_v3beta1.GetChangelogRequest(name='name_value')
    response = client.get_changelog(request=request)
    print(response)