import re
import create_fileset_entry

def test_create_fileset_entry(capsys, client, random_entry_name):
    if False:
        while True:
            i = 10
    entry_name_pattern = '(?P<entry_group_name>.+?)/entries/(?P<entry_id>.+?$)'
    entry_name_matches = re.match(entry_name_pattern, random_entry_name)
    entry_group_name = entry_name_matches.group('entry_group_name')
    entry_id = entry_name_matches.group('entry_id')
    create_fileset_entry.create_fileset_entry(client, entry_group_name, entry_id)
    (out, err) = capsys.readouterr()
    assert 'Created entry {}'.format(random_entry_name) in out