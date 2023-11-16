"""Helpers: Download: should_try_releases."""

def test_base(repository):
    if False:
        while True:
            i = 10
    repository.ref = 'dummy'
    repository.data.category = 'plugin'
    repository.data.releases = True
    assert repository.should_try_releases

def test_ref_is_default(repository):
    if False:
        return 10
    repository.ref = 'main'
    repository.data.category = 'plugin'
    repository.data.releases = True
    assert not repository.should_try_releases

def test_category_is_wrong(repository):
    if False:
        for i in range(10):
            print('nop')
    repository.ref = 'dummy'
    repository.data.category = 'integration'
    repository.data.releases = True
    assert not repository.should_try_releases

def test_no_releases(repository):
    if False:
        return 10
    repository.ref = 'dummy'
    repository.data.category = 'plugin'
    repository.data.releases = False
    assert not repository.should_try_releases

def test_zip_release(repository):
    if False:
        print('Hello World!')
    repository.data.releases = False
    repository.repository_manifest.zip_release = True
    repository.repository_manifest.filename = 'test.zip'
    assert repository.should_try_releases
    repository.ref = 'main'
    assert not repository.should_try_releases