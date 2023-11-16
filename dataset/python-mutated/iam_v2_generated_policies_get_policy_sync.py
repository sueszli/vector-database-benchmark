from google.cloud import iam_v2

def sample_get_policy():
    if False:
        return 10
    client = iam_v2.PoliciesClient()
    request = iam_v2.GetPolicyRequest(name='name_value')
    response = client.get_policy(request=request)
    print(response)