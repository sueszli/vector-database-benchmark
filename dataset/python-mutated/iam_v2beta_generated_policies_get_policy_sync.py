from google.cloud import iam_v2beta

def sample_get_policy():
    if False:
        while True:
            i = 10
    client = iam_v2beta.PoliciesClient()
    request = iam_v2beta.GetPolicyRequest(name='name_value')
    response = client.get_policy(request=request)
    print(response)