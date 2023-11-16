from google.cloud import iam_v2
from google.cloud.iam_v2 import Policy, types

def get_deny_policy(project_id: str, policy_id: str) -> Policy:
    if False:
        i = 10
        return i + 15
    '\n    Retrieve the deny policy given the project ID and policy ID.\n\n    project_id: ID or number of the Google Cloud project you want to use.\n    policy_id: The ID of the deny policy you want to retrieve.\n    '
    policies_client = iam_v2.PoliciesClient()
    attachment_point = f'cloudresourcemanager.googleapis.com%2Fprojects%2F{project_id}'
    request = types.GetPolicyRequest()
    request.name = f'policies/{attachment_point}/denypolicies/{policy_id}'
    policy = policies_client.get_policy(request=request)
    print(f'Retrieved the deny policy: {policy_id} : {policy}')
    return policy
if __name__ == '__main__':
    import uuid
    project_id = 'your-google-cloud-project-id'
    policy_id = f'deny-{uuid.uuid4()}'
    policy = get_deny_policy(project_id, policy_id)