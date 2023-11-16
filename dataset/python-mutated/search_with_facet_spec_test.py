import re
import subprocess
from google.api_core.retry import Retry
from search_with_facet_spec import search

@Retry()
def test_search_with_filtering_pass():
    if False:
        print('Hello World!')
    output = str(subprocess.check_output('python search_with_facet_spec.py', shell=True))
    assert re.match('.*search request.*', output)
    assert re.match('.*search response.*', output)
    assert re.match('.*results.*id.*', output)
    assert re.match('.*facets.*?colorFamilies.*', output)

def test_search_with_filtering():
    if False:
        print('Hello World!')
    response = search()
    assert len(response.results) == 10
    product_title = response.results[0].product.title
    assert re.match('.*Tee.*', product_title)
    assert response.facets[0].key == 'colorFamilies'