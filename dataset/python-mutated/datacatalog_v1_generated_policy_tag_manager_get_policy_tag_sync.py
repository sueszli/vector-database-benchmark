from google.cloud import datacatalog_v1

def sample_get_policy_tag():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1.PolicyTagManagerClient()
    request = datacatalog_v1.GetPolicyTagRequest(name='name_value')
    response = client.get_policy_tag(request=request)
    print(response)