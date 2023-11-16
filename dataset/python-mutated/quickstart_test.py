import quickstart

def test_quickstart(capsys):
    if False:
        i = 10
        return i + 15
    quickstart.run_quickstart()
    (out, _) = capsys.readouterr()
    assert 'Saved' in out