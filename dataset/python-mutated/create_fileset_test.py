import create_fileset

def test_create_fileset(capsys, client, project_id, random_entry_group_id, random_entry_id, resources_to_delete):
    if False:
        for i in range(10):
            print('nop')
    location = 'us-central1'
    override_values = {'project_id': project_id, 'fileset_entry_group_id': random_entry_group_id, 'fileset_entry_id': random_entry_id}
    expected_group_name = client.entry_group_path(project_id, location, random_entry_group_id)
    expected_entry_name = client.entry_path(project_id, location, random_entry_group_id, random_entry_id)
    create_fileset.create_fileset(override_values)
    (out, err) = capsys.readouterr()
    assert f'Created entry group: {expected_group_name}' in out
    assert f'Created fileset entry: {expected_entry_name}' in out
    resources_to_delete['entry_groups'].append(expected_group_name)
    resources_to_delete['entries'].append(expected_entry_name)