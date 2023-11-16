import re
import subprocess
from google.api_core.retry import Retry
from search_with_pagination import search

@Retry()
def test_search_with_pagination_pass():
    if False:
        print('Hello World!')
    output = str(subprocess.check_output('python search_with_pagination.py', shell=True))
    assert re.match('.*search request.*', output)
    assert re.match('.*search response.*', output)
    assert re.match('.*results.*id.*', output)

def test_search_with_pagination():
    if False:
        i = 10
        return i + 15
    response = search()
    assert len(response.results) == 6