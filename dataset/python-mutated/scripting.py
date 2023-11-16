from pyramid.config import global_registries
from pyramid.exceptions import ConfigurationError
from pyramid.interfaces import IRequestFactory, IRootFactory
from pyramid.request import Request, apply_request_extensions
from pyramid.threadlocal import RequestContext
from pyramid.traversal import DefaultRootFactory

def get_root(app, request=None):
    if False:
        print('Hello World!')
    "Return a tuple composed of ``(root, closer)`` when provided a\n    :term:`router` instance as the ``app`` argument.  The ``root``\n    returned is the application root object.  The ``closer`` returned\n    is a callable (accepting no arguments) that should be called when\n    your scripting application is finished using the root.\n\n    ``request`` is passed to the :app:`Pyramid` application root\n    factory to compute the root. If ``request`` is None, a default\n    will be constructed using the registry's :term:`Request Factory`\n    via the :meth:`pyramid.interfaces.IRequestFactory.blank` method.\n    "
    registry = app.registry
    if request is None:
        request = _make_request('/', registry)
    request.registry = registry
    ctx = RequestContext(request)
    ctx.begin()

    def closer():
        if False:
            while True:
                i = 10
        ctx.end()
    root = app.root_factory(request)
    return (root, closer)

def prepare(request=None, registry=None):
    if False:
        print('Hello World!')
    "This function pushes data onto the Pyramid threadlocal stack\n    (request and registry), making those objects 'current'.  It\n    returns a dictionary useful for bootstrapping a Pyramid\n    application in a scripting environment.\n\n    ``request`` is passed to the :app:`Pyramid` application root\n    factory to compute the root. If ``request`` is None, a default\n    will be constructed using the registry's :term:`Request Factory`\n    via the :meth:`pyramid.interfaces.IRequestFactory.blank` method.\n\n    If ``registry`` is not supplied, the last registry loaded from\n    :attr:`pyramid.config.global_registries` will be used. If you\n    have loaded more than one :app:`Pyramid` application in the\n    current process, you may not want to use the last registry\n    loaded, thus you can search the ``global_registries`` and supply\n    the appropriate one based on your own criteria.\n\n    The function returns a dictionary composed of ``root``,\n    ``closer``, ``registry``, ``request`` and ``root_factory``.  The\n    ``root`` returned is the application's root resource object.  The\n    ``closer`` returned is a callable (accepting no arguments) that\n    should be called when your scripting application is finished\n    using the root.  ``registry`` is the resolved registry object.\n    ``request`` is the request object passed or the constructed request\n    if no request is passed.  ``root_factory`` is the root factory used\n    to construct the root.\n\n    This function may be used as a context manager to call the ``closer``\n    automatically:\n\n    .. code-block:: python\n\n       registry = config.registry\n       with prepare(registry) as env:\n           request = env['request']\n           # ...\n\n    .. versionchanged:: 1.8\n\n       Added the ability to use the return value as a context manager.\n\n    .. versionchanged:: 2.0\n\n       Request finished callbacks added via\n       :meth:`pyramid.request.Request.add_finished_callback` will be invoked\n       by the ``closer``.\n\n    "
    if registry is None:
        registry = getattr(request, 'registry', global_registries.last)
    if registry is None:
        raise ConfigurationError('No valid Pyramid applications could be found, make sure one has been created before trying to activate it.')
    if request is None:
        request = _make_request('/', registry)
    request.registry = registry
    ctx = RequestContext(request)
    ctx.begin()
    apply_request_extensions(request)

    def closer():
        if False:
            print('Hello World!')
        if request.finished_callbacks:
            request._process_finished_callbacks()
        ctx.end()
    root_factory = registry.queryUtility(IRootFactory, default=DefaultRootFactory)
    root = root_factory(request)
    if getattr(request, 'context', None) is None:
        request.context = root
    return AppEnvironment(root=root, closer=closer, registry=registry, request=request, root_factory=root_factory)

class AppEnvironment(dict):

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, type, value, traceback):
        if False:
            i = 10
            return i + 15
        self['closer']()

def _make_request(path, registry=None):
    if False:
        print('Hello World!')
    "Return a :meth:`pyramid.request.Request` object anchored at a\n    given path. The object returned will be generated from the supplied\n    registry's :term:`Request Factory` using the\n    :meth:`pyramid.interfaces.IRequestFactory.blank` method.\n\n    This request object can be passed to :meth:`pyramid.scripting.get_root`\n    or :meth:`pyramid.scripting.prepare` to initialize an application in\n    preparation for executing a script with a proper environment setup.\n    URLs can then be generated with the object, as well as rendering\n    templates.\n\n    If ``registry`` is not supplied, the last registry loaded from\n    :attr:`pyramid.config.global_registries` will be used. If you have\n    loaded more than one :app:`Pyramid` application in the current\n    process, you may not want to use the last registry loaded, thus\n    you can search the ``global_registries`` and supply the appropriate\n    one based on your own criteria.\n    "
    if registry is None:
        registry = global_registries.last
    request_factory = registry.queryUtility(IRequestFactory, default=Request)
    request = request_factory.blank(path)
    request.registry = registry
    return request