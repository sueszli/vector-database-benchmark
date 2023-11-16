import create_entry_group

def test_create_entry_group(capsys, client, project_id, random_entry_group_id):
    if False:
        print('Hello World!')
    create_entry_group.create_entry_group(project_id, random_entry_group_id)
    (out, err) = capsys.readouterr()
    assert 'Created entry group projects/{}/locations/{}/entryGroups/{}'.format(project_id, 'us-central1', random_entry_group_id) in out