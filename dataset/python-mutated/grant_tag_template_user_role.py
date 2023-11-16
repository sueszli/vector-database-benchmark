def grant_tag_template_user_role(override_values):
    if False:
        return 10
    'Grants a user the Tag Template User role for a given template.'
    from google.cloud import datacatalog_v1
    from google.iam.v1 import iam_policy_pb2 as iam_policy
    from google.iam.v1 import policy_pb2
    datacatalog = datacatalog_v1.DataCatalogClient()
    project_id = 'project_id'
    tag_template_id = 'existing_tag_template_id'
    member_id = 'user:super-cool.test-user@gmail.com'
    project_id = override_values.get('project_id', project_id)
    tag_template_id = override_values.get('tag_template_id', tag_template_id)
    member_id = override_values.get('member_id', member_id)
    location = 'us-central1'
    template_name = datacatalog_v1.DataCatalogClient.tag_template_path(project_id, location, tag_template_id)
    policy = datacatalog.get_iam_policy(resource=template_name)
    binding = policy_pb2.Binding()
    binding.role = 'roles/datacatalog.tagTemplateUser'
    binding.members.append(member_id)
    policy.bindings.append(binding)
    set_policy_request = iam_policy.SetIamPolicyRequest(resource=template_name, policy=policy)
    policy = datacatalog.set_iam_policy(set_policy_request)
    for binding in policy.bindings:
        for member in binding.members:
            print(f'Member: {member}, Role: {binding.role}')