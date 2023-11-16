import quickstart_get_saved_query

def test_get_saved_query(capsys, test_saved_query):
    if False:
        for i in range(10):
            print('nop')
    quickstart_get_saved_query.get_saved_query(test_saved_query.name)
    (out, _) = capsys.readouterr()
    assert 'gotten_saved_query' in out