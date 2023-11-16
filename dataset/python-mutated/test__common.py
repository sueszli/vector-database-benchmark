import pytest
from reactpy import html
from reactpy.backend._common import CommonOptions, traversal_safe_path, vdom_head_elements_to_html

def test_common_options_url_prefix_starts_with_slash():
    if False:
        i = 10
        return i + 15
    CommonOptions(url_prefix='')
    with pytest.raises(ValueError, match="start with '/'"):
        CommonOptions(url_prefix='not-start-withslash')

@pytest.mark.parametrize('bad_path', ['../escaped', 'ok/../../escaped', 'ok/ok-again/../../ok-yet-again/../../../escaped'])
def test_catch_unsafe_relative_path_traversal(tmp_path, bad_path):
    if False:
        print('Hello World!')
    with pytest.raises(ValueError, match='Unsafe path'):
        traversal_safe_path(tmp_path, *bad_path.split('/'))

@pytest.mark.parametrize('vdom_in, html_out', [('<title>example</title>', '<title>example</title>'), ('<head></head>', '<head></head>'), (html.head(html.meta({'charset': 'utf-8'}), html.title('example')), '<meta charset="utf-8"><title>example</title>'), (html._(html.meta({'charset': 'utf-8'}), html.title('example')), '<meta charset="utf-8"><title>example</title>'), ([html.meta({'charset': 'utf-8'}), html.title('example')], '<meta charset="utf-8"><title>example</title>')])
def test_vdom_head_elements_to_html(vdom_in, html_out):
    if False:
        i = 10
        return i + 15
    assert vdom_head_elements_to_html(vdom_in) == html_out