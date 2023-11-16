import io
import pretend
import pytest
from pyramid.httpexceptions import HTTPBadRequest, HTTPMovedPermanently
from pyramid.request import Request
from pyramid.response import Response
from warehouse import sanity

class TestJunkEncoding:

    def test_valid(self):
        if False:
            for i in range(10):
                print('nop')
        request = Request({'QUERY_STRING': ':action=browse', 'PATH_INFO': '/pypi'})
        sanity.junk_encoding(request)

    def test_invalid_qsl(self):
        if False:
            for i in range(10):
                print('nop')
        request = Request({'QUERY_STRING': '%Aaction=browse'})
        with pytest.raises(HTTPBadRequest, match='Invalid bytes in query string.'):
            sanity.junk_encoding(request)

    def test_invalid_path(self):
        if False:
            for i in range(10):
                print('nop')
        request = Request({'PATH_INFO': '/projects/abouÅt'})
        with pytest.raises(HTTPBadRequest, match='Invalid bytes in URL.'):
            sanity.junk_encoding(request)

class TestInvalidForms:

    def test_valid(self):
        if False:
            print('Hello World!')
        request = Request({'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': 'multipart/form-data; boundary=c397e2aa2980f1a53dee37c05b8fb45a', 'wsgi.input': io.BytesIO(b'--------------------------c397e2aa2980f1a53dee37c05b8fb45a\r\nContent-Disposition: form-data; name="person"\r\nanonymous')})
        sanity.invalid_forms(request)

    def test_invalid_form(self):
        if False:
            for i in range(10):
                print('nop')
        request = Request({'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': 'multipart/form-data', 'wsgi.input': io.BytesIO(b'Content-Disposition: form-data; name="person"\r\nanonymous')})
        with pytest.raises(HTTPBadRequest, match='Invalid Form Data.'):
            sanity.invalid_forms(request)

    def test_not_post(self):
        if False:
            return 10
        request = Request({'REQUEST_METHOD': 'GET'})
        sanity.invalid_forms(request)

@pytest.mark.parametrize(('original_location', 'expected_location'), [('/a/path/to/nowhere', '/a/path/to/nowhere'), ('/project/☃/', '/project/%E2%98%83/'), (None, None)])
def test_unicode_redirects(original_location, expected_location):
    if False:
        while True:
            i = 10
    if original_location:
        resp_in = HTTPMovedPermanently(original_location)
    else:
        resp_in = Response()
    resp_out = sanity.unicode_redirects(resp_in)
    assert resp_out.location == expected_location

class TestSanityTween:

    def test_ingress_valid(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')
        junk_encoding = pretend.call_recorder(lambda request: None)
        monkeypatch.setattr(sanity, 'junk_encoding', junk_encoding)
        invalid_forms = pretend.call_recorder(lambda request: None)
        monkeypatch.setattr(sanity, 'invalid_forms', invalid_forms)
        response = pretend.stub()
        handler = pretend.call_recorder(lambda request: response)
        registry = pretend.stub()
        request = pretend.stub()
        tween = sanity.sanity_tween_factory_ingress(handler, registry)
        assert tween(request) is response
        assert junk_encoding.calls == [pretend.call(request)]
        assert invalid_forms.calls == [pretend.call(request)]
        assert handler.calls == [pretend.call(request)]

    def test_ingress_invalid(self, monkeypatch):
        if False:
            while True:
                i = 10
        response = HTTPBadRequest()

        @pretend.call_recorder
        def junk_encoding(request):
            if False:
                for i in range(10):
                    print('nop')
            raise response
        monkeypatch.setattr(sanity, 'junk_encoding', junk_encoding)
        handler = pretend.call_recorder(lambda request: response)
        registry = pretend.stub()
        request = pretend.stub()
        tween = sanity.sanity_tween_factory_ingress(handler, registry)
        assert tween(request) is response
        assert junk_encoding.calls == [pretend.call(request)]
        assert handler.calls == []

    def test_egress(self, monkeypatch):
        if False:
            while True:
                i = 10
        unicode_redirects = pretend.call_recorder(lambda resp: resp)
        monkeypatch.setattr(sanity, 'unicode_redirects', unicode_redirects)
        response = pretend.stub()
        handler = pretend.call_recorder(lambda request: response)
        registry = pretend.stub()
        request = pretend.stub()
        tween = sanity.sanity_tween_factory_egress(handler, registry)
        assert tween(request) is response
        assert handler.calls == [pretend.call(request)]
        assert unicode_redirects.calls == [pretend.call(response)]