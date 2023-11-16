from google.cloud import datacatalog_v1

def sample_delete_policy_tag():
    if False:
        while True:
            i = 10
    client = datacatalog_v1.PolicyTagManagerClient()
    request = datacatalog_v1.DeletePolicyTagRequest(name='name_value')
    client.delete_policy_tag(request=request)