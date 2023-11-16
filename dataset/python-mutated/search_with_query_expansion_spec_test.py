import re
import subprocess
from google.api_core.retry import Retry
from search_with_query_expansion_spec import search

@Retry()
def test_search_with_query_expansion_spec_pass():
    if False:
        for i in range(10):
            print('nop')
    output = str(subprocess.check_output('python search_with_query_expansion_spec.py', shell=True))
    assert re.match('.*search request.*', output)
    assert re.match('.*search response.*', output)
    assert re.match('.*results.*id.*', output)

def test_search_with_query_expansion_spec():
    if False:
        return 10
    response = search()
    assert response.query_expansion_info.expanded_query is True