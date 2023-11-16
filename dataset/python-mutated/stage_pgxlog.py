import pytest

class PgXlog(object):
    """Test utility for staging a pg_xlog directory."""

    def __init__(self, cluster):
        if False:
            while True:
                i = 10
        self.cluster = cluster
        self.pg_xlog = cluster.join('pg_xlog')
        self.pg_xlog.ensure(dir=True)
        self.status = self.pg_xlog.join('archive_status')
        self.status.ensure(dir=True)

    def touch(self, name, status):
        if False:
            while True:
                i = 10
        assert status in ('.ready', '.done')
        self.pg_xlog.join(name).ensure(file=True)
        self.status.join(name + status).ensure(file=True)

    def seg(self, name):
        if False:
            print('Hello World!')
        return self.pg_xlog.join(name)

    def assert_exists(self, name, status):
        if False:
            print('Hello World!')
        assert status in ('.ready', '.done')
        assert self.pg_xlog.join(name).check(exists=1)
        assert self.status.join(name + status).check(exists=1)

@pytest.fixture()
def pg_xlog(tmpdir, monkeypatch):
    if False:
        while True:
            i = 10
    'Set up xlog utility functions and change directories.'
    monkeypatch.chdir(tmpdir)
    return PgXlog(tmpdir)