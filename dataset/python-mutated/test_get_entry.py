import get_entry

def test_get_entry(client, entry):
    if False:
        return 10
    name = client.parse_entry_path(entry)
    retrieved_entry = get_entry.sample_get_entry(name['project'], name['location'], name['entry_group'], name['entry'])
    assert retrieved_entry.name == entry