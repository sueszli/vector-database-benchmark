import create_custom_entry

def test_create_custom_entry(capsys, client, project_id, random_entry_group_id, random_entry_id, resources_to_delete):
    if False:
        print('Hello World!')
    location = 'us-central1'
    override_values = {'project_id': project_id, 'entry_id': random_entry_id, 'entry_group_id': random_entry_group_id}
    expected_entry_group = client.entry_group_path(project_id, location, random_entry_group_id)
    expected_entry = client.entry_path(project_id, location, random_entry_group_id, random_entry_id)
    create_custom_entry.create_custom_entry(override_values)
    (out, err) = capsys.readouterr()
    assert f'Created entry group: {expected_entry_group}' in out
    assert f'Created entry: {expected_entry}' in out
    resources_to_delete['entries'].append(expected_entry)