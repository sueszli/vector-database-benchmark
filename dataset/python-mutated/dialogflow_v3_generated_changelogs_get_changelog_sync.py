from google.cloud import dialogflowcx_v3

def sample_get_changelog():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.ChangelogsClient()
    request = dialogflowcx_v3.GetChangelogRequest(name='name_value')
    response = client.get_changelog(request=request)
    print(response)