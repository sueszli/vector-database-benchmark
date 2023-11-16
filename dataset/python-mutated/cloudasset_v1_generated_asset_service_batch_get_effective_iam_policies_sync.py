from google.cloud import asset_v1

def sample_batch_get_effective_iam_policies():
    if False:
        return 10
    client = asset_v1.AssetServiceClient()
    request = asset_v1.BatchGetEffectiveIamPoliciesRequest(scope='scope_value', names=['names_value1', 'names_value2'])
    response = client.batch_get_effective_iam_policies(request=request)
    print(response)