def list_deny_policy(project_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    from google.cloud import iam_v2
    from google.cloud.iam_v2 import types
    '\n    List all the deny policies that are attached to a resource.\n    A resource can have up to 5 deny policies.\n\n    project_id: ID or number of the Google Cloud project you want to use.\n    '
    policies_client = iam_v2.PoliciesClient()
    attachment_point = f'cloudresourcemanager.googleapis.com%2Fprojects%2F{project_id}'
    request = types.ListPoliciesRequest()
    request.parent = f'policies/{attachment_point}/denypolicies'
    policies = policies_client.list_policies(request=request)
    for policy in policies:
        print(policy.name)
    print('Listed all deny policies')
if __name__ == '__main__':
    import uuid
    project_id = 'your-google-cloud-project-id'
    policy_id = f'deny-{uuid.uuid4()}'
    list_deny_policy(project_id)