from google.cloud import metastore_v1alpha

def sample_remove_iam_policy():
    if False:
        return 10
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.RemoveIamPolicyRequest(resource='resource_value')
    response = client.remove_iam_policy(request=request)
    print(response)