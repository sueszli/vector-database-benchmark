from google.cloud import metastore_v1beta

def sample_remove_iam_policy():
    if False:
        while True:
            i = 10
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.RemoveIamPolicyRequest(resource='resource_value')
    response = client.remove_iam_policy(request=request)
    print(response)