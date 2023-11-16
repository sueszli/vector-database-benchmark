import set_endpoint

def test_set_endpoint(capsys):
    if False:
        print('Hello World!')
    set_endpoint.set_endpoint()
    (out, _) = capsys.readouterr()
    assert 'System' in out
    assert 'bounds:' in out