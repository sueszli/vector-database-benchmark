import os
import pytest
import spack.config
import spack.error
import spack.fetch_strategy
import spack.stage

@pytest.mark.parametrize('_fetch_method', ['curl', 'urllib'])
def test_gcsfetchstrategy_without_url(_fetch_method):
    if False:
        for i in range(10):
            print('nop')
    'Ensure constructor with no URL fails.'
    with spack.config.override('config:url_fetch_method', _fetch_method):
        with pytest.raises(ValueError):
            spack.fetch_strategy.GCSFetchStrategy(None)

@pytest.mark.parametrize('_fetch_method', ['curl', 'urllib'])
def test_gcsfetchstrategy_bad_url(tmpdir, _fetch_method):
    if False:
        i = 10
        return i + 15
    'Ensure fetch with bad URL fails as expected.'
    testpath = str(tmpdir)
    with spack.config.override('config:url_fetch_method', _fetch_method):
        fetcher = spack.fetch_strategy.GCSFetchStrategy(url='file:///does-not-exist')
        assert fetcher is not None
        with spack.stage.Stage(fetcher, path=testpath) as stage:
            assert stage is not None
            assert fetcher.archive_file is None
            with pytest.raises(spack.error.FetchError):
                fetcher.fetch()

@pytest.mark.parametrize('_fetch_method', ['curl', 'urllib'])
def test_gcsfetchstrategy_downloaded(tmpdir, _fetch_method):
    if False:
        return 10
    'Ensure fetch with archive file already downloaded is a noop.'
    testpath = str(tmpdir)
    archive = os.path.join(testpath, 'gcs.tar.gz')
    with spack.config.override('config:url_fetch_method', _fetch_method):

        class Archived_GCSFS(spack.fetch_strategy.GCSFetchStrategy):

            @property
            def archive_file(self):
                if False:
                    i = 10
                    return i + 15
                return archive
        url = 'gcs:///{0}'.format(archive)
        fetcher = Archived_GCSFS(url=url)
        with spack.stage.Stage(fetcher, path=testpath):
            fetcher.fetch()