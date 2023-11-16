from google.cloud import config_v1

def sample_get_revision():
    if False:
        i = 10
        return i + 15
    client = config_v1.ConfigClient()
    request = config_v1.GetRevisionRequest(name='name_value')
    response = client.get_revision(request=request)
    print(response)