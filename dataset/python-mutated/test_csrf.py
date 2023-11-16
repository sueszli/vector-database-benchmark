import pretend
import pytest
from pyramid.httpexceptions import HTTPMethodNotAllowed
from pyramid.viewderivers import INGRESS, csrf_view
from warehouse import csrf

class TestRequireMethodView:

    def test_passes_through_on_falsey(self):
        if False:
            return 10
        view = pretend.stub()
        info = pretend.stub(options={'require_methods': False})
        assert csrf.require_method_view(view, info) is view

    @pytest.mark.parametrize('method', ['GET', 'HEAD', 'OPTIONS'])
    def test_allows_safe_by_default(self, method):
        if False:
            for i in range(10):
                print('nop')
        response = pretend.stub()

        @pretend.call_recorder
        def view(context, request):
            if False:
                for i in range(10):
                    print('nop')
            return response
        info = pretend.stub(options={})
        wrapped_view = csrf.require_method_view(view, info)
        context = pretend.stub()
        request = pretend.stub(method=method)
        assert wrapped_view(context, request) is response
        assert view.calls == [pretend.call(context, request)]

    @pytest.mark.parametrize('method', ['POST', 'PUT', 'DELETE'])
    def test_disallows_unsafe_by_default(self, method):
        if False:
            i = 10
            return i + 15

        @pretend.call_recorder
        def view(context, request):
            if False:
                print('Hello World!')
            pass
        info = pretend.stub(options={})
        wrapped_view = csrf.require_method_view(view, info)
        context = pretend.stub()
        request = pretend.stub(method=method)
        with pytest.raises(HTTPMethodNotAllowed):
            wrapped_view(context, request)
        assert view.calls == []

    def test_allows_passing_other_methods(self):
        if False:
            i = 10
            return i + 15
        response = pretend.stub()

        @pretend.call_recorder
        def view(context, request):
            if False:
                i = 10
                return i + 15
            return response
        info = pretend.stub(options={'require_methods': ['POST']})
        wrapped_view = csrf.require_method_view(view, info)
        context = pretend.stub()
        request = pretend.stub(method='POST')
        assert wrapped_view(context, request) is response
        assert view.calls == [pretend.call(context, request)]

    def test_allows_exception_views_by_default(self):
        if False:
            for i in range(10):
                print('nop')
        response = pretend.stub()

        @pretend.call_recorder
        def view(context, request):
            if False:
                i = 10
                return i + 15
            return response
        info = pretend.stub(options={})
        wrapped_view = csrf.require_method_view(view, info)
        context = pretend.stub()
        request = pretend.stub(method='POST', exception=pretend.stub())
        assert wrapped_view(context, request) is response
        assert view.calls == [pretend.call(context, request)]

    def test_explicit_controls_exception_views(self):
        if False:
            while True:
                i = 10

        @pretend.call_recorder
        def view(context, request):
            if False:
                while True:
                    i = 10
            pass
        info = pretend.stub(options={'require_methods': ['POST']})
        wrapped_view = csrf.require_method_view(view, info)
        context = pretend.stub()
        request = pretend.stub(method='GET')
        with pytest.raises(HTTPMethodNotAllowed):
            wrapped_view(context, request)
        assert view.calls == []

def test_includeme():
    if False:
        return 10
    config = pretend.stub(set_default_csrf_options=pretend.call_recorder(lambda **kw: None), add_view_deriver=pretend.call_recorder(lambda *args, **kw: None))
    csrf.includeme(config)
    assert config.set_default_csrf_options.calls == [pretend.call(require_csrf=True)]
    assert config.add_view_deriver.calls == [pretend.call(csrf_view, under=INGRESS, over='secured_view'), pretend.call(csrf.require_method_view, under=INGRESS, over='csrf_view')]