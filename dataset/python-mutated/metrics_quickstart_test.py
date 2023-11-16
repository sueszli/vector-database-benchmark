import metrics_quickstart

def test_quickstart_main(capsys):
    if False:
        return 10
    metrics_quickstart.main()
    output = capsys.readouterr()
    assert 'Fake latency recorded' in output.out