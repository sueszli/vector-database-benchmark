from __future__ import annotations
import pytest
pytest
import bs4
import bokeh.embed.server as bes

@pytest.fixture
def test_plot() -> None:
    if False:
        i = 10
        return i + 15
    from bokeh.plotting import figure
    test_plot = figure()
    test_plot.circle([1, 2], [2, 3])
    return test_plot

class TestServerDocument:

    def test_invalid_resources_param(self) -> None:
        if False:
            return 10
        with pytest.raises(ValueError):
            bes.server_document(url='http://localhost:8081/foo/bar/sliders', resources=123)
        with pytest.raises(ValueError):
            bes.server_document(url='http://localhost:8081/foo/bar/sliders', resources='whatever')

    def test_resources_default_is_implicit(self) -> None:
        if False:
            print('Hello World!')
        r = bes.server_document(url='http://localhost:8081/foo/bar/sliders', resources='default')
        assert 'resources=' not in r

    def test_resources_none(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        r = bes.server_document(url='http://localhost:8081/foo/bar/sliders', resources=None)
        assert 'resources=none' in r

    def test_general(self) -> None:
        if False:
            print('Hello World!')
        url = 'http://localhost:8081/foo/bar/sliders'
        r = bes.server_document(url=url)
        assert 'bokeh-app-path=/foo/bar/sliders' in r
        assert 'bokeh-absolute-url=http://localhost:8081/foo/bar/sliders' in r
        html = bs4.BeautifulSoup(r, 'html.parser')
        scripts = html.findAll(name='script')
        assert len(scripts) == 1
        script = scripts[0]
        attrs = script.attrs
        assert list(attrs) == ['id']
        divid = attrs['id']
        request = f'''xhr.open('GET', "{url}/autoload.js?bokeh-autoload-element={divid}&bokeh-app-path=/foo/bar/sliders&bokeh-absolute-url={url}", true);'''
        assert request in script.string

    def test_script_attrs_arguments_provided(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        url = 'http://localhost:5006'
        r = bes.server_document(arguments=dict(foo=10))
        assert 'foo=10' in r
        html = bs4.BeautifulSoup(r, 'html.parser')
        scripts = html.findAll(name='script')
        assert len(scripts) == 1
        script = scripts[0]
        attrs = script.attrs
        assert list(attrs) == ['id']
        divid = attrs['id']
        request = f'''xhr.open('GET', "{url}/autoload.js?bokeh-autoload-element={divid}&bokeh-absolute-url={url}&foo=10", true);'''
        assert request in script.string

    def test_script_attrs_url_provided_absolute_resources(self) -> None:
        if False:
            return 10
        url = 'http://localhost:8081/foo/bar/sliders'
        r = bes.server_document(url=url)
        assert 'bokeh-app-path=/foo/bar/sliders' in r
        assert 'bokeh-absolute-url=http://localhost:8081/foo/bar/sliders' in r
        html = bs4.BeautifulSoup(r, 'html.parser')
        scripts = html.findAll(name='script')
        assert len(scripts) == 1
        script = scripts[0]
        attrs = script.attrs
        assert list(attrs) == ['id']
        divid = attrs['id']
        request = f'''xhr.open('GET', "{url}/autoload.js?bokeh-autoload-element={divid}&bokeh-app-path=/foo/bar/sliders&bokeh-absolute-url={url}", true);'''
        assert request in script.string

    def test_script_attrs_url_provided(self) -> None:
        if False:
            i = 10
            return i + 15
        url = 'http://localhost:8081/foo/bar/sliders'
        r = bes.server_document(url=url, relative_urls=True)
        assert 'bokeh-app-path=/foo/bar/sliders' in r
        html = bs4.BeautifulSoup(r, 'html.parser')
        scripts = html.findAll(name='script')
        assert len(scripts) == 1
        script = scripts[0]
        attrs = script.attrs
        assert list(attrs) == ['id']
        divid = attrs['id']
        request = f'''xhr.open('GET', "{url}/autoload.js?bokeh-autoload-element={divid}&bokeh-app-path=/foo/bar/sliders", true);'''
        assert request in script.string

class TestServerSession:

    def test_return_type(self, test_plot) -> None:
        if False:
            i = 10
            return i + 15
        r = bes.server_session(test_plot, session_id='fakesession')
        assert isinstance(r, str)

    def test_script_attrs_session_id_provided(self, test_plot) -> None:
        if False:
            print('Hello World!')
        url = 'http://localhost:5006'
        r = bes.server_session(test_plot, session_id='fakesession')
        html = bs4.BeautifulSoup(r, 'html.parser')
        scripts = html.findAll(name='script')
        assert len(scripts) == 1
        script = scripts[0]
        attrs = script.attrs
        assert list(attrs) == ['id']
        divid = attrs['id']
        request = f'''xhr.open('GET', "{url}/autoload.js?bokeh-autoload-element={divid}&bokeh-absolute-url={url}", true);'''
        assert request in script.string
        assert 'xhr.setRequestHeader("Bokeh-Session-Id", "fakesession")' in script.string

    def test_invalid_resources_param(self, test_plot) -> None:
        if False:
            print('Hello World!')
        with pytest.raises(ValueError):
            bes.server_session(test_plot, session_id='fakesession', resources=123)
        with pytest.raises(ValueError):
            bes.server_session(test_plot, session_id='fakesession', resources='whatever')

    def test_resources_default_is_implicit(self, test_plot) -> None:
        if False:
            i = 10
            return i + 15
        r = bes.server_session(test_plot, session_id='fakesession', resources='default')
        assert 'resources=' not in r

    def test_resources_none(self, test_plot) -> None:
        if False:
            while True:
                i = 10
        r = bes.server_session(test_plot, session_id='fakesession', resources=None)
        assert 'resources=none' in r

    def test_model_none(self) -> None:
        if False:
            print('Hello World!')
        url = 'http://localhost:5006'
        r = bes.server_session(None, session_id='fakesession')
        html = bs4.BeautifulSoup(r, 'html.parser')
        scripts = html.findAll(name='script')
        assert len(scripts) == 1
        script = scripts[0]
        attrs = script.attrs
        assert list(attrs) == ['id']
        divid = attrs['id']
        request = f'{url}/autoload.js?bokeh-autoload-element={divid}&bokeh-absolute-url={url}'
        assert request in script.string
        assert 'xhr.setRequestHeader("Bokeh-Session-Id", "fakesession")' in script.string

    def test_general(self, test_plot) -> None:
        if False:
            return 10
        url = 'http://localhost:5006'
        r = bes.server_session(test_plot, session_id='fakesession')
        html = bs4.BeautifulSoup(r, 'html.parser')
        scripts = html.findAll(name='script')
        assert len(scripts) == 1
        script = scripts[0]
        attrs = script.attrs
        assert list(attrs) == ['id']
        divid = attrs['id']
        request = f'''xhr.open('GET', "{url}/autoload.js?bokeh-autoload-element={divid}&bokeh-absolute-url={url}", true);'''
        assert request in script.string
        assert 'xhr.setRequestHeader("Bokeh-Session-Id", "fakesession")' in script.string

class Test__clean_url:

    def test_default(self) -> None:
        if False:
            print('Hello World!')
        assert bes._clean_url('default') == bes.DEFAULT_SERVER_HTTP_URL.rstrip('/')

    def test_bad_ws(self) -> None:
        if False:
            return 10
        with pytest.raises(ValueError):
            bes._clean_url('ws://foo')

    def test_arg(self) -> None:
        if False:
            return 10
        assert bes._clean_url('http://foo/bar') == 'http://foo/bar'
        assert bes._clean_url('http://foo/bar/') == 'http://foo/bar'

class Test__get_app_path:

    def test_arg(self) -> None:
        if False:
            i = 10
            return i + 15
        assert bes._get_app_path('foo') == '/foo'
        assert bes._get_app_path('http://foo') == '/'
        assert bes._get_app_path('http://foo/bar') == '/bar'
        assert bes._get_app_path('https://foo') == '/'
        assert bes._get_app_path('https://foo/bar') == '/bar'

class Test__process_arguments:

    def test_None(self) -> None:
        if False:
            i = 10
            return i + 15
        assert bes._process_arguments(None) == ''

    def test_args(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        args = dict(foo=10, bar='baz')
        r = bes._process_arguments(args)
        assert r == '&foo=10&bar=baz' or r == '&bar=baz&foo=10'

    def test_args_ignores_bokeh_prefixed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        args = dict(foo=10, bar='baz')
        args['bokeh-junk'] = 20
        r = bes._process_arguments(args)
        assert r == '&foo=10&bar=baz' or r == '&bar=baz&foo=10'

class Test__process_app_path:

    def test_root(self) -> None:
        if False:
            print('Hello World!')
        assert bes._process_app_path('/') == ''

    def test_arg(self) -> None:
        if False:
            print('Hello World!')
        assert bes._process_app_path('/stuff') == '&bokeh-app-path=/stuff'

class Test__process_relative_urls:

    def test_True(self) -> None:
        if False:
            while True:
                i = 10
        assert bes._process_relative_urls(True, '') == ''
        assert bes._process_relative_urls(True, '/stuff') == ''

    def test_Flase(self) -> None:
        if False:
            print('Hello World!')
        assert bes._process_relative_urls(False, '/stuff') == '&bokeh-absolute-url=/stuff'

class Test__process_resources:

    def test_bad_input(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError):
            bes._process_resources('foo')

    def test_None(self) -> None:
        if False:
            return 10
        assert bes._process_resources(None) == '&resources=none'

    def test_default(self) -> None:
        if False:
            print('Hello World!')
        assert bes._process_resources('default') == ''

def Test__src_path(object):
    if False:
        for i in range(10):
            print('nop')

    def test_args(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert bes._src_path('http://foo', '1234') == 'http://foo/autoload.js?bokeh-autoload-element=1234'