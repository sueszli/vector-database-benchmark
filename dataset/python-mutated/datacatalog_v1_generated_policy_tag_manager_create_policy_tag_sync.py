from google.cloud import datacatalog_v1

def sample_create_policy_tag():
    if False:
        while True:
            i = 10
    client = datacatalog_v1.PolicyTagManagerClient()
    request = datacatalog_v1.CreatePolicyTagRequest(parent='parent_value')
    response = client.create_policy_tag(request=request)
    print(response)