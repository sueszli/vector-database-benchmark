from google.cloud import datacatalog_v1beta1

def sample_get_policy_tag():
    if False:
        print('Hello World!')
    client = datacatalog_v1beta1.PolicyTagManagerClient()
    request = datacatalog_v1beta1.GetPolicyTagRequest(name='name_value')
    response = client.get_policy_tag(request=request)
    print(response)