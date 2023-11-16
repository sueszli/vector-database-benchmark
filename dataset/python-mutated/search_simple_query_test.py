import re
import subprocess
from google.api_core.retry import Retry
from search_simple_query import search

@Retry()
def test_search_simple_query_pass():
    if False:
        while True:
            i = 10
    output = str(subprocess.check_output('python search_simple_query.py', shell=True))
    assert re.match('.*search request.*', output)
    assert re.match('.*search response.*', output)
    assert re.match('.*results.*id.*', output)

def test_search_simple_query_response():
    if False:
        return 10
    response = search()
    assert len(response.results) == 10