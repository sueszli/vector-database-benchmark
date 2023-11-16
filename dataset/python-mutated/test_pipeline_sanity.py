import pytest
from wal_e import pipeline
from wal_e import pipebuf

def create_bogus_payload(dirname):
    if False:
        return 10
    payload = b'abcd' * 1048576
    payload_file = dirname.join('payload')
    payload_file.write(payload)
    return (payload, payload_file)

def test_rate_limit(tmpdir):
    if False:
        return 10
    (payload, payload_file) = create_bogus_payload(tmpdir)
    with open(str(payload_file)) as f:
        pl = pipeline.PipeViewerRateLimitFilter(1048576 * 100, stdin=f)
        pl.start()
        round_trip = pl.stdout.read()
        pl.finish()
        assert round_trip == payload

def test_upload_download_pipeline(tmpdir, rate_limit):
    if False:
        print('Hello World!')
    (payload, payload_file) = create_bogus_payload(tmpdir)
    test_upload = tmpdir.join('upload')
    with open(str(test_upload), 'wb') as upload:
        with open(str(payload_file), 'rb') as inp:
            with pipeline.get_upload_pipeline(inp, upload, rate_limit=rate_limit):
                pass
    with open(str(test_upload), 'rb') as completed:
        round_trip = completed.read()
    test_download = tmpdir.join('download')
    with open(str(test_upload), 'rb') as upload:
        with open(str(test_download), 'wb') as download:
            with pipeline.get_download_pipeline(upload, download):
                pass
    with open(str(test_download), 'rb') as completed:
        round_trip = completed.read()
    assert round_trip == payload

def test_close_process_when_normal():
    if False:
        while True:
            i = 10
    'Process leaks must not occur in successful cases'
    with pipeline.get_cat_pipeline(pipeline.PIPE, pipeline.PIPE) as pl:
        assert len(pl.commands) == 1
        assert pl.commands[0]._process.poll() is None
    pipeline_wait(pl)

def test_close_process_when_exception():
    if False:
        i = 10
        return i + 15
    'Process leaks must not occur when an exception is raised'
    exc = Exception('boom')
    with pytest.raises(Exception) as e:
        with pipeline.get_cat_pipeline(pipeline.PIPE, pipeline.PIPE) as pl:
            assert len(pl.commands) == 1
            assert pl.commands[0]._process.poll() is None
            raise exc
    assert e.value is exc
    pipeline_wait(pl)

def test_close_process_when_aborted():
    if False:
        while True:
            i = 10
    'Process leaks must not occur when the pipeline is aborted'
    with pipeline.get_cat_pipeline(pipeline.PIPE, pipeline.PIPE) as pl:
        assert len(pl.commands) == 1
        assert pl.commands[0]._process.poll() is None
        pl.abort()
    pipeline_wait(pl)

def test_double_close():
    if False:
        print('Hello World!')
    'A file should is able to be closed twice without raising'
    with pipeline.get_cat_pipeline(pipeline.PIPE, pipeline.PIPE) as pl:
        assert isinstance(pl.stdin, pipebuf.NonBlockBufferedWriter)
        assert not pl.stdin.closed
        pl.stdin.close()
        assert pl.stdin.closed
        pl.stdin.close()
        assert isinstance(pl.stdout, pipebuf.NonBlockBufferedReader)
        assert not pl.stdout.closed
        pl.stdout.close()
        assert pl.stdout.closed
        pl.stdout.close()
    pipeline_wait(pl)

def pipeline_wait(pl):
    if False:
        return 10
    for command in pl.commands:
        command.wait()

def pytest_generate_tests(metafunc):
    if False:
        print('Hello World!')
    if 'rate_limit' in metafunc.funcargnames:
        metafunc.parametrize('rate_limit', [None, int(2 ** 25)])