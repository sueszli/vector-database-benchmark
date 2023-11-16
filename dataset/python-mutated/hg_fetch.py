import os
import pytest
from llnl.util.filesystem import mkdirp, touch, working_dir
import spack.config
import spack.repo
from spack.fetch_strategy import HgFetchStrategy
from spack.spec import Spec
from spack.stage import Stage
from spack.util.executable import which
from spack.version import Version
pytestmark = [pytest.mark.skipif(not which('hg'), reason='requires mercurial to be installed'), pytest.mark.not_on_windows('Failing on Win')]

@pytest.mark.parametrize('type_of_test', ['default', 'rev0'])
@pytest.mark.parametrize('secure', [True, False])
def test_fetch(type_of_test, secure, mock_hg_repository, config, mutable_mock_repo, monkeypatch):
    if False:
        print('Hello World!')
    "Tries to:\n\n    1. Fetch the repo using a fetch strategy constructed with\n       supplied args (they depend on type_of_test).\n    2. Check if the test_file is in the checked out repository.\n    3. Assert that the repository is at the revision supplied.\n    4. Add and remove some files, then reset the repo, and\n       ensure it's all there again.\n    "
    t = mock_hg_repository.checks[type_of_test]
    h = mock_hg_repository.hash
    s = Spec('hg-test').concretized()
    monkeypatch.setitem(s.package.versions, Version('hg'), t.args)
    with s.package.stage:
        with spack.config.override('config:verify_ssl', secure):
            s.package.do_stage()
        with working_dir(s.package.stage.source_path):
            assert h() == t.revision
            file_path = os.path.join(s.package.stage.source_path, t.file)
            assert os.path.isdir(s.package.stage.source_path)
            assert os.path.isfile(file_path)
            os.unlink(file_path)
            assert not os.path.isfile(file_path)
            untracked_file = 'foobarbaz'
            touch(untracked_file)
            assert os.path.isfile(untracked_file)
            s.package.do_restage()
            assert not os.path.isfile(untracked_file)
            assert os.path.isdir(s.package.stage.source_path)
            assert os.path.isfile(file_path)
            assert h() == t.revision

def test_hg_extra_fetch(tmpdir):
    if False:
        print('Hello World!')
    'Ensure a fetch after expanding is effectively a no-op.'
    testpath = str(tmpdir)
    fetcher = HgFetchStrategy(hg='file:///not-a-real-hg-repo')
    with Stage(fetcher, path=testpath) as stage:
        source_path = stage.source_path
        mkdirp(source_path)
        fetcher.fetch()