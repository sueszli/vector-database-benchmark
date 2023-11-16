from google.cloud import datacatalog_v1

def sample_update_policy_tag():
    if False:
        while True:
            i = 10
    client = datacatalog_v1.PolicyTagManagerClient()
    request = datacatalog_v1.UpdatePolicyTagRequest()
    response = client.update_policy_tag(request=request)
    print(response)