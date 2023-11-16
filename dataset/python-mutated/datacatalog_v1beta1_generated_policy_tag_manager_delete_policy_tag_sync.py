from google.cloud import datacatalog_v1beta1

def sample_delete_policy_tag():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1beta1.PolicyTagManagerClient()
    request = datacatalog_v1beta1.DeletePolicyTagRequest(name='name_value')
    client.delete_policy_tag(request=request)