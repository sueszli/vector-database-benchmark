import re
import subprocess
from google.api_core.retry import Retry
from search_with_ordering import search

@Retry()
def test_search_with_ordering_pass():
    if False:
        return 10
    output = str(subprocess.check_output('python search_with_ordering.py', shell=True))
    assert re.match('.*search request.*', output)
    assert re.match('.*search response.*', output)
    assert re.match('.*results.*id.*', output)

def test_search_with_ordering():
    if False:
        while True:
            i = 10
    response = search()
    assert len(response.results) == 10
    assert response.results[0].product.price_info.price == 39