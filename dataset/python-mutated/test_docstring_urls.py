import importlib
import pkgutil
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import urlopen
import pytest
import modin.pandas

@pytest.fixture
def doc_urls(get_generated_doc_urls):
    if False:
        for i in range(10):
            print('nop')
    for modinfo in pkgutil.walk_packages(modin.pandas.__path__, 'modin.pandas.'):
        try:
            importlib.import_module(modinfo.name)
        except ModuleNotFoundError:
            pass
    return sorted(get_generated_doc_urls())

def test_all_urls_exist(doc_urls):
    if False:
        i = 10
        return i + 15
    broken = []

    def _test_url(url):
        if False:
            print('Hello World!')
        try:
            with urlopen(url):
                pass
        except HTTPError:
            broken.append(url)
    with ThreadPoolExecutor(32) as pool:
        pool.map(_test_url, doc_urls)
    assert not broken, 'Invalid URLs detected'