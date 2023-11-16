import re
import subprocess
from google.api_core.retry import Retry
from search_with_boost_spec import search

@Retry()
def test_search_with_boost_spec_pass():
    if False:
        i = 10
        return i + 15
    output = str(subprocess.check_output('python search_with_boost_spec.py', shell=True))
    assert re.match('.*search request.*', output)
    assert re.match('.*search response.*', output)
    assert re.match('.*results.*id.*', output)

def test_search_with_boost_spec():
    if False:
        while True:
            i = 10
    response = search()
    assert len(response.results) == 10
    product_title = response.results[0].product.title
    assert re.match('.*Tee.*', product_title)