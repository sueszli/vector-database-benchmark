import base64
import glob
import os
import subprocess
from html.parser import HTMLParser
from urllib.parse import unquote
import nbconvert
import pytest
from selenium.common.exceptions import UnexpectedAlertPresentException
from folium.utilities import temp_html_filepath

def find_notebooks():
    if False:
        i = 10
        return i + 15
    'Return a list of filenames of the example notebooks.'
    path = os.path.dirname(__file__)
    pattern = os.path.join(path, '..', '..', 'docs', '**', '*.md')
    files = glob.glob(pattern, recursive=True)
    if files:
        return files
    else:
        raise OSError('Could not find the notebooks')

@pytest.mark.parametrize('filepath', find_notebooks())
def test_notebook(filepath, driver):
    if False:
        print('Hello World!')
    if 'WmsTimeDimension' in filepath:
        pytest.xfail('WmsTimeDimension.ipynb external resource makes this test flaky')
    for filepath_html in get_notebook_html(filepath):
        driver.get_file(filepath_html)
        try:
            assert driver.wait_until('.folium-map')
        except UnexpectedAlertPresentException:
            print('skipping', filepath_html, 'because of alert')
            continue
        driver.verify_js_logs()

def get_notebook_html(filepath_notebook):
    if False:
        print('Hello World!')
    'Convert markdown to notebook to html files, remove them when done.'
    subprocess.run(['jupytext', '--to', 'notebook', '--execute', filepath_notebook])
    filepath_notebook = filepath_notebook.replace('.md', '.ipynb')
    html_exporter = nbconvert.HTMLExporter()
    (body, _) = html_exporter.from_filename(filepath_notebook)
    parser = IframeParser()
    parser.feed(body)
    iframes = parser.iframes
    for iframe in iframes:
        with temp_html_filepath(iframe) as filepath_html:
            yield filepath_html

class IframeParser(HTMLParser):
    """Extract the iframes from an html page."""

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.iframes = []

    def handle_starttag(self, tag, attrs):
        if False:
            print('Hello World!')
        if tag == 'iframe':
            attrs = dict(attrs)
            if 'srcdoc' in attrs:
                html_bytes = attrs['srcdoc'].encode()
            elif 'data-html' in attrs:
                data_html = attrs['data-html']
                if '%' in data_html[:20]:
                    html_bytes = unquote(data_html).encode()
                else:
                    html_bytes = base64.b64decode(data_html)
            else:
                src = attrs['src']
                html_base64 = src.split(',')[-1]
                html_bytes = base64.b64decode(html_base64)
            self.iframes.append(html_bytes)