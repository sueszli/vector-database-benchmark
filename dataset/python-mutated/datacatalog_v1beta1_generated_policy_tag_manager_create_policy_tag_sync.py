from google.cloud import datacatalog_v1beta1

def sample_create_policy_tag():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1beta1.PolicyTagManagerClient()
    request = datacatalog_v1beta1.CreatePolicyTagRequest(parent='parent_value')
    response = client.create_policy_tag(request=request)
    print(response)