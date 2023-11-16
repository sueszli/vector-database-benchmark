def update_deny_policy(project_id: str, policy_id: str, etag: str) -> None:
    if False:
        print('Hello World!')
    from google.cloud import iam_v2
    from google.cloud.iam_v2 import types
    '\n    Update the deny rules and/ or its display name after policy creation.\n\n    project_id: ID or number of the Google Cloud project you want to use.\n\n    policy_id: The ID of the deny policy you want to retrieve.\n\n    etag: Etag field that identifies the policy version. The etag changes each time\n    you update the policy. Get the etag of an existing policy by performing a GetPolicy request.\n    '
    policies_client = iam_v2.PoliciesClient()
    attachment_point = f'cloudresourcemanager.googleapis.com%2Fprojects%2F{project_id}'
    deny_rule = types.DenyRule()
    deny_rule.denied_principals = ['principalSet://goog/public:all']
    deny_rule.denied_permissions = ['cloudresourcemanager.googleapis.com/projects.delete']
    deny_rule.denial_condition = {'expression': "!resource.matchTag('12345678/env', 'prod')"}
    policy_rule = types.PolicyRule()
    policy_rule.description = 'block all principals from deleting projects, unless the principal is a member of project-admins@example.com and the project being deleted has a tag with the value prod'
    policy_rule.deny_rule = deny_rule
    policy = types.Policy()
    policy.name = f'policies/{attachment_point}/denypolicies/{policy_id}'
    policy.etag = etag
    policy.rules = [policy_rule]
    request = types.UpdatePolicyRequest()
    request.policy = policy
    result = policies_client.update_policy(request=request).result()
    print(f"Updated the deny policy: {result.name.rsplit('/')[-1]}")
if __name__ == '__main__':
    import uuid
    project_id = 'your-google-cloud-project-id'
    policy_id = f'deny-{uuid.uuid4()}'
    etag = 'etag'
    update_deny_policy(project_id, policy_id, etag)