import re
import subprocess
from google.api_core.retry import Retry
from search_with_filtering import search

@Retry()
def test_search_with_filtering_pass():
    if False:
        i = 10
        return i + 15
    output = str(subprocess.check_output('python search_with_filtering.py', shell=True))
    assert re.match('.*search request.*', output)
    assert re.match('.*search response.*', output)
    assert re.match('.*results.*id.*', output)

def test_search_with_filtering():
    if False:
        for i in range(10):
            print('nop')
    response = search()
    assert len(response.results) == 10
    product_title = response.results[0].product.title
    assert re.match('.*Tee.*', product_title)
    assert re.match('.*Black.*', product_title)
    assert 'Black' in response.results[0].product.color_info.color_families