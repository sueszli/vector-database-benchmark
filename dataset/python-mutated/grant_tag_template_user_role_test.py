import grant_tag_template_user_role

def test_grant_tag_template_user_role(capsys, project_id, random_existing_tag_template_id, valid_member_id):
    if False:
        for i in range(10):
            print('nop')
    override_values = {'project_id': project_id, 'tag_template_id': random_existing_tag_template_id, 'member_id': valid_member_id}
    grant_tag_template_user_role.grant_tag_template_user_role(override_values)
    (out, err) = capsys.readouterr()
    assert f'Member: {valid_member_id}, Role: roles/datacatalog.tagTemplateUser' in out