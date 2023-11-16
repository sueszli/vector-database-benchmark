from typing import Any, Callable, NoReturn, Optional
from flask import Flask, current_app, g, request
from flask.wrappers import Request, Response
from oso import Oso, OsoError
from werkzeug.exceptions import Forbidden
from .context import _app_context

class FlaskOso:
    """oso flask plugin

    This plugin must be initialized with a flask app, either using the
    ``app`` parameter in the constructor, or by calling :py:meth:`init_app` after
    construction.

    The plugin must be initialized with an :py:class:`oso.Oso` instance before
    use, either by passing one to the constructor or calling
    :py:meth:`set_oso`.

    **Authorization**

    - :py:meth:`FlaskOso.authorize`: Check whether an actor, action and resource is
      authorized. Integrates with flask to provide defaults for actor & action.

    **Configuration**

    - :py:meth:`require_authorization`: Require at least one
      :py:meth:`FlaskOso.authorize` call for every request.
    - :py:meth:`set_get_actor`: Override how oso determines the actor
      associated with a request if none is provided to :py:meth:`FlaskOso.authorize`.
    - :py:meth:`set_unauthorized_action`: Control how :py:meth:`FlaskOso.authorize`
      handles an unauthorized request.
    - :py:meth:`perform_route_authorization`:
      Call `authorize(resource=flask.request)` before every request.
    """
    _app: Optional[Flask]
    _oso: Optional[Oso]

    def __init__(self, oso: Optional[Oso]=None, app: Optional[Flask]=None) -> None:
        if False:
            print('Hello World!')
        self._app = app
        self._oso = None

        def unauthorized() -> NoReturn:
            if False:
                while True:
                    i = 10
            raise Forbidden('Unauthorized')
        self._unauthorized_action = unauthorized
        self._get_actor = lambda : g.current_user
        if self._app is not None:
            self.init_app(self._app)
        if oso is not None:
            self.set_oso(oso)

    def set_oso(self, oso: Oso) -> None:
        if False:
            print('Hello World!')
        'Set the oso instance to use for authorization\n\n        Must be called if ``oso`` is not provided to the constructor.\n        '
        if oso == self._oso:
            return
        self._oso = oso
        self._oso.register_class(Request)

    def init_app(self, app: Flask) -> None:
        if False:
            while True:
                i = 10
        "Initialize ``app`` for use with this instance of ``FlaskOso``.\n\n        Must be called if ``app`` isn't provided to the constructor.\n        "
        app.teardown_appcontext(self.teardown)
        app.before_request(self._provide_oso)

    def set_get_actor(self, func: Callable[[], Any]) -> None:
        if False:
            print('Hello World!')
        'Provide a function that oso will use to get the current actor.\n\n        :param func: A function to call with no parameters to get the actor if\n                     it is not provided to :py:meth:`FlaskOso.authorize`. The return value\n                     is used as the actor.\n        '
        self._get_actor = func

    def set_unauthorized_action(self, func: Callable[[], Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set a function that will be called to handle an authorization failure.\n\n        The default behavior is to raise a Forbidden exception, returning a 403\n        response.\n\n        :param func: A function to call with no parameters when a request is\n                     not authorized.\n        '
        self._unauthorized_action = func

    def require_authorization(self, app: Optional[Flask]=None) -> None:
        if False:
            while True:
                i = 10
        'Enforce authorization on every request to ``app``.\n\n        :param app: The app to require authorization for. Can be omitted if\n                    the ``app`` parameter was used in the ``FlaskOso``\n                    constructor.\n\n        If :py:meth:`FlaskOso.authorize` is not called during the request processing,\n        raises an :py:class:`oso.OsoError`.\n\n        Call :py:meth:`FlaskOso.skip_authorization` to skip this check for a particular\n        request.\n        '
        if app is None:
            app = self._app
        if app is None:
            raise OsoError('Cannot require authorization without Flask app object')
        app.after_request(self._require_authorization)

    def perform_route_authorization(self, app: Optional[Flask]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Perform route authorization before every request.\n\n        Route authorization will call :py:meth:`oso.Oso.is_allowed` with the\n        current request (from ``flask.request``) as the resource and the method\n        (from ``flask.request.method``) as the action.\n\n        :param app: The app to require authorization for. Can be omitted if\n                    the ``app`` parameter was used in the ``FlaskOso``\n                    constructor.\n        '
        if app is None:
            app = self._app
        if app is None:
            raise OsoError('Cannot perform route authorization without Flask app object')
        app.before_request(self._perform_route_authorization)

    def skip_authorization(self, reason: Optional[str]=None) -> None:
        if False:
            return 10
        'Opt-out of authorization for the current request.\n\n        Will prevent ``require_authorization`` from causing an error.\n\n        See also: :py:func:`flask_oso.skip_authorization` for a route decorator version.\n        '
        _authorize_called()

    def authorize(self, resource: Any, *, actor: Optional[Any]=None, action: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Check whether the current request should be allowed.\n\n        Calls :py:meth:`oso.Oso.is_allowed` to check authorization. If a request\n        is unauthorized, raises a ``werkzeug.exceptions.Forbidden``\n        exception.  This behavior can be controlled with\n        :py:meth:`set_unauthorized_action`.\n\n        :param actor: The actor to authorize. Defaults to ``flask.g.current_user``.\n                      Use :py:meth:`set_get_actor` to override.\n        :param action: The action to authorize. Defaults to\n                       ``flask.request.method``.\n        :param resource: The resource to authorize.  The flask request object\n                         (``flask.request``) can be passed to authorize a\n                         request based on route path or other request properties.\n\n        See also: :py:func:`flask_oso.authorize` for a route decorator version.\n        '
        if actor is None:
            try:
                actor = self.current_actor
            except AttributeError as e:
                raise OsoError('Getting the current actor failed. You may need to override the current actor function with FlaskOso#set_get_actor') from e
        if action is None:
            action = request.method
        if resource is request:
            resource = request._get_current_object()
        if self.oso is None:
            raise OsoError('Cannot perform authorization without oso instance')
        allowed = self.oso.is_allowed(actor, action, resource)
        _authorize_called()
        if not allowed:
            self._unauthorized_action()

    @property
    def app(self) -> Flask:
        if False:
            print('Hello World!')
        return self._app or current_app

    @property
    def oso(self) -> Optional[Oso]:
        if False:
            for i in range(10):
                print('nop')
        return self._oso

    @property
    def current_actor(self) -> Any:
        if False:
            i = 10
            return i + 15
        return self._get_actor()

    def _provide_oso(self) -> None:
        if False:
            while True:
                i = 10
        top = _app_context()
        if not hasattr(top, 'oso_flask_oso'):
            top.oso_flask_oso = self

    def _perform_route_authorization(self) -> None:
        if False:
            print('Hello World!')
        if not request.url_rule:
            return
        self.authorize(resource=request)

    def _require_authorization(self, response: Response) -> Response:
        if False:
            i = 10
            return i + 15
        if not request.url_rule:
            return response
        if not getattr(_app_context(), 'oso_flask_authorize_called', False):
            raise OsoError('Authorize not called.')
        return response

    def teardown(self, exception):
        if False:
            for i in range(10):
                print('nop')
        pass

def _authorize_called() -> None:
    if False:
        i = 10
        return i + 15
    'Mark current request as authorized.'
    _app_context().oso_flask_authorize_called = True