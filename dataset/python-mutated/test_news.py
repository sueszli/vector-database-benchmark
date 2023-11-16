"""Testing that news entries are well formed."""
import os
import re
import pytest
NEWSDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'news')
CATEGORIES = frozenset(['Added', 'Changed', 'Deprecated', 'Removed', 'Fixed', 'Security'])
single_grave_reg = re.compile('[^`]`[^`]+`[^`_]')

def check_news_file(fname):
    if False:
        for i in range(10):
            print('nop')
    import restructuredtext_lint
    name = fname.name
    with open(fname.path) as f:
        content = f.read()
    errors = restructuredtext_lint.lint(content)
    if errors:
        err_msgs = os.linesep.join((err.message for err in errors))
        pytest.fail(f'{fname}: Invalid ReST\n{err_msgs}')
    form = ''
    for (i, l) in enumerate(content.splitlines()):
        if l.startswith('**'):
            cat = l[2:].rsplit(':')[0]
            if cat not in CATEGORIES:
                pytest.fail(f'{name}:{i + 1}: {cat!r} not a proper category must be one of {list(CATEGORIES)}', pytrace=True)
            if l.endswith('None'):
                form += '3'
            else:
                form += '2'
        elif l.startswith('* <news item>'):
            form += '4'
        elif l.startswith('* ') or l.startswith('- ') or l.startswith('  '):
            form += '1'
        elif l.strip() == '':
            form += '0'
        else:
            pytest.fail(f'{name}:{i + 1}: invalid rst', pytrace=True)
    reg = re.compile('^(3(0|$)|20(1|4)(1|0|4)*0|204$)+$')
    if not reg.match(form):
        print(form)
        pytest.fail(f'{name}: invalid rst', pytrace=True)

@pytest.fixture(params=list(os.scandir(NEWSDIR)))
def fname(request):
    if False:
        print('Hello World!')
    if request.node.config.option.markexpr != 'news':
        pytest.skip('Run news items check explicitly')
    return request.param

@pytest.mark.news
def test_news(fname):
    if False:
        for i in range(10):
            print('nop')
    (base, ext) = os.path.splitext(fname.path)
    assert 'rst' in ext
    check_news_file(fname)