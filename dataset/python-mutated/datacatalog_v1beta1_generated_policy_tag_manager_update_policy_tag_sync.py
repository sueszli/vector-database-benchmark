from google.cloud import datacatalog_v1beta1

def sample_update_policy_tag():
    if False:
        print('Hello World!')
    client = datacatalog_v1beta1.PolicyTagManagerClient()
    request = datacatalog_v1beta1.UpdatePolicyTagRequest()
    response = client.update_policy_tag(request=request)
    print(response)