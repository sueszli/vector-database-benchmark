from warehouse.utils import readme

def test_render_with_none():
    if False:
        print('Hello World!')
    result = readme.render(None)
    assert result is None

def test_can_render_rst():
    if False:
        i = 10
        return i + 15
    result = readme.render('raw thing', 'text/x-rst')
    assert result == '<p>raw thing</p>\n'

def test_cant_render_rst():
    if False:
        while True:
            i = 10
    result = readme.render('raw `<thing', 'text/x-rst')
    assert result == 'raw `&lt;thing'

def test_can_render_plaintext():
    if False:
        i = 10
        return i + 15
    result = readme.render('raw thing', 'text/plain')
    assert result == '<pre>raw thing</pre>'

def test_can_render_markdown():
    if False:
        print('Hello World!')
    result = readme.render('raw thing', 'text/markdown')
    assert result == '<p>raw thing</p>\n'

def test_can_render_missing_content_type():
    if False:
        while True:
            i = 10
    result = readme.render('raw thing')
    assert result == '<p>raw thing</p>\n'

def test_can_render_blank_content_type():
    if False:
        print('Hello World!')
    result = readme.render('wild thing', '')
    assert result == '<p>wild thing</p>\n'

def test_renderer_version():
    if False:
        while True:
            i = 10
    assert readme.renderer_version() is not None