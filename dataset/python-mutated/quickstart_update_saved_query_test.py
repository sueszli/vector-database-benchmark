import quickstart_update_saved_query

def test_update_saved_query(capsys, test_saved_query):
    if False:
        i = 10
        return i + 15
    quickstart_update_saved_query.update_saved_query(test_saved_query.name, 'bar')
    (out, _) = capsys.readouterr()
    assert 'updated_saved_query' in out