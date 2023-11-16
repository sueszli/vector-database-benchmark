"""
Utilities for testing nameko services.
"""
import inspect
from collections import OrderedDict
from contextlib import contextmanager
import eventlet
from eventlet import event
from mock import MagicMock
from nameko.exceptions import ExtensionNotFound
from nameko.extensions import DependencyProvider, Entrypoint
from nameko.testing.utils import get_extension
from nameko.testing.waiting import WaitResult, wait_for_call

@contextmanager
def entrypoint_hook(container, method_name, context_data=None, timeout=30):
    if False:
        i = 10
        return i + 15
    ' Yield a function providing an entrypoint into a hosted service.\n\n    The yielded function may be called as if it were the bare method defined\n    in the service class. Intended to be used as an integration testing\n    utility.\n\n    :Parameters:\n        container : ServiceContainer\n            The container hosting the service owning the entrypoint\n        method_name : str\n            The name of the entrypoint decorated method on the service class\n        context_data : dict\n            Context data to provide for the call, e.g. a language, auth\n            token or session.\n        timeout : int\n            Maximum seconds to wait\n\n    **Usage**\n\n    To verify that `ServiceX` and `ServiceY` are compatible, make an\n    integration test that checks their interaction:\n\n    .. literalinclude:: ../examples/testing/integration_x_y_test.py\n\n    '
    entrypoint = get_extension(container, Entrypoint, method_name=method_name)
    if entrypoint is None:
        raise ExtensionNotFound("No entrypoint for '{}' found on container {}.".format(method_name, container))

    def hook(*args, **kwargs):
        if False:
            print('Hello World!')
        hook_result = event.Event()

        def wait_for_entrypoint():
            if False:
                i = 10
                return i + 15
            try:
                with entrypoint_waiter(container, method_name, timeout=timeout) as waiter_result:
                    container.spawn_worker(entrypoint, args, kwargs, context_data=context_data)
                hook_result.send(waiter_result.get())
            except Exception as exc:
                hook_result.send_exception(exc)

        def wait_for_container():
            if False:
                return 10
            try:
                container.wait()
            except Exception as exc:
                if not hook_result.ready():
                    hook_result.send_exception(exc)
        eventlet.spawn_n(wait_for_entrypoint)
        eventlet.spawn_n(wait_for_container)
        return hook_result.wait()
    yield hook

@contextmanager
def entrypoint_waiter(container, method_name, timeout=30, callback=None):
    if False:
        while True:
            i = 10
    ' Context manager that waits until an entrypoint has fired, and\n    the generated worker has exited and been torn down.\n\n    It yields a :class:`nameko.testing.waiting.WaitResult` object that can be\n    used to get the result returned (exception raised) by the entrypoint\n    after the waiter has exited.\n\n    :Parameters:\n        container : ServiceContainer\n            The container hosting the service owning the entrypoint\n        method_name : str\n            The name of the entrypoint decorated method on the service class\n        timeout : int\n            Maximum seconds to wait\n        callback : callable\n            Function to conditionally control whether the entrypoint_waiter\n            should exit for a particular invocation\n\n    The `timeout` argument specifies the maximum number of seconds the\n    `entrypoint_waiter` should wait before exiting. It can be disabled by\n    passing `None`. The default is 30 seconds.\n\n    Optionally allows a `callback` to be provided which is invoked whenever\n    the entrypoint fires. If provided, the callback must return `True`\n    for the `entrypoint_waiter` to exit. The signature for the callback\n    function is::\n\n        def callback(worker_ctx, result, exc_info):\n            pass\n\n    Where there parameters are as follows:\n\n        worker_ctx (WorkerContext): WorkerContext of the entrypoint call.\n\n        result (object): The return value of the entrypoint.\n\n        exc_info (tuple): Tuple as returned by `sys.exc_info` if the\n            entrypoint raised an exception, otherwise `None`.\n\n    **Usage**\n\n    ::\n        class Service(object):\n            name = "service"\n\n            @event_handler(\'srcservice\', \'eventtype\')\n            def handle_event(self, msg):\n                return msg\n\n        container = ServiceContainer(Service, config)\n        container.start()\n\n        # basic\n        with entrypoint_waiter(container, \'handle_event\'):\n            ...  # action that dispatches event\n\n        # giving access to the result\n        with entrypoint_waiter(container, \'handle_event\') as result:\n            ...  # action that dispatches event\n        res = result.get()\n\n        # with custom timeout\n        with entrypoint_waiter(container, \'handle_event\', timeout=5):\n            ...  # action that dispatches event\n\n        # with callback that waits until entrypoint stops raising\n        def callback(worker_ctx, result, exc_info):\n            if exc_info is None:\n                return True\n\n        with entrypoint_waiter(container, \'handle_event\', callback=callback):\n            ...  # action that dispatches event\n\n    '
    if not get_extension(container, Entrypoint, method_name=method_name):
        raise RuntimeError('{} has no entrypoint `{}`'.format(container.service_name, method_name))

    class Result(WaitResult):
        worker_ctx = None

        def send(self, worker_ctx, result, exc_info):
            if False:
                return 10
            self.worker_ctx = worker_ctx
            super(Result, self).send(result, exc_info)
    waiter_callback = callback
    waiter_result = Result()

    def on_worker_result(worker_ctx, result, exc_info):
        if False:
            for i in range(10):
                print('nop')
        complete = False
        if worker_ctx.entrypoint.method_name == method_name:
            if not callable(waiter_callback):
                complete = True
            else:
                complete = waiter_callback(worker_ctx, result, exc_info)
        if complete:
            waiter_result.send(worker_ctx, result, exc_info)
        return complete

    def on_worker_teardown(worker_ctx):
        if False:
            for i in range(10):
                print('nop')
        if waiter_result.worker_ctx is worker_ctx:
            return True
        return False
    exc = entrypoint_waiter.Timeout('Timeout on {}.{} after {} seconds'.format(container.service_name, method_name, timeout))
    with eventlet.Timeout(timeout, exception=exc):
        with wait_for_call(container, '_worker_teardown', lambda args, kwargs, res, exc: on_worker_teardown(*args)):
            with wait_for_call(container, '_worker_result', lambda args, kwargs, res, exc: on_worker_result(*args)):
                yield waiter_result

class EntrypointWaiterTimeout(Exception):
    pass
entrypoint_waiter.Timeout = EntrypointWaiterTimeout

def worker_factory(service_cls, **dependencies):
    if False:
        return 10
    ' Return an instance of ``service_cls`` with its injected dependencies\n    replaced with :class:`~mock.MagicMock` objects, or as given in\n    ``dependencies``.\n\n    **Usage**\n\n    The following example service proxies calls to a "maths" service via\n    an ``RpcProxy`` dependency::\n\n        from nameko.rpc import RpcProxy, rpc\n\n        class ConversionService(object):\n            name = "conversions"\n\n            maths_rpc = RpcProxy("maths")\n\n            @rpc\n            def inches_to_cm(self, inches):\n                return self.maths_rpc.multiply(inches, 2.54)\n\n            @rpc\n            def cm_to_inches(self, cms):\n                return self.maths_rpc.divide(cms, 2.54)\n\n    Use the ``worker_factory`` to create an instance of\n    ``ConversionService`` with its dependencies replaced by MagicMock objects::\n\n        service = worker_factory(ConversionService)\n\n    Nameko\'s entrypoints do not modify the service methods, so instance methods\n    can be called directly with the same signature. The replaced dependencies\n    can be used as any other MagicMock object, so a complete unit test for\n    the conversion service may look like this::\n\n        # create worker instance\n        service = worker_factory(ConversionService)\n\n        # replace "maths" service\n        service.maths_rpc.multiply.side_effect = lambda x, y: x * y\n        service.maths_rpc.divide.side_effect = lambda x, y: x / y\n\n        # test inches_to_cm business logic\n        assert service.inches_to_cm(300) == 762\n        service.maths_rpc.multiply.assert_called_once_with(300, 2.54)\n\n        # test cms_to_inches business logic\n        assert service.cms_to_inches(762) == 300\n        service.maths_rpc.divide.assert_called_once_with(762, 2.54)\n\n    *Providing Dependencies*\n\n    The ``**dependencies`` kwargs to ``worker_factory`` can be used to provide\n    a replacement dependency instead of a mock. For example, to unit test a\n    service against a real database:\n\n    .. literalinclude::\n        ../examples/testing/alternative_dependency_unit_test.py\n\n    If a named dependency provider does not exist on ``service_cls``, a\n    ``ExtensionNotFound`` exception is raised.\n\n    '
    service = service_cls()
    for (name, attr) in inspect.getmembers(service_cls):
        if isinstance(attr, DependencyProvider):
            try:
                dependency = dependencies.pop(name)
            except KeyError:
                dependency = MagicMock()
            setattr(service, name, dependency)
    if dependencies:
        raise ExtensionNotFound("DependencyProvider(s) '{}' not found on {}.".format(dependencies.keys(), service_cls))
    return service

class MockDependencyProvider(DependencyProvider):

    def __init__(self, attr_name, dependency=None):
        if False:
            i = 10
            return i + 15
        self.attr_name = attr_name
        self.dependency = MagicMock() if dependency is None else dependency

    def get_dependency(self, worker_ctx):
        if False:
            while True:
                i = 10
        return self.dependency

def _replace_dependencies(container, **dependency_map):
    if False:
        for i in range(10):
            print('nop')
    if container.started:
        raise RuntimeError('You must replace dependencies before the container is started.')
    dependency_names = {dep.attr_name for dep in container.dependencies}
    missing = set(dependency_map) - dependency_names
    if missing:
        raise ExtensionNotFound("Dependency(s) '{}' not found on {}.".format(missing, container))
    existing_providers = {dep.attr_name: dep for dep in container.dependencies if dep.attr_name in dependency_map}
    for (name, replacement) in dependency_map.items():
        existing_provider = existing_providers[name]
        replacement_provider = MockDependencyProvider(name, dependency=replacement)
        container.dependencies.remove(existing_provider)
        container.dependencies.add(replacement_provider)

def replace_dependencies(container, *dependencies, **dependency_map):
    if False:
        while True:
            i = 10
    ' Replace the dependency providers on ``container`` with\n    instances of :class:`MockDependencyProvider`.\n\n    Dependencies named in *dependencies will be replaced with a\n    :class:`MockDependencyProvider`, which injects a MagicMock instead of the\n    dependency.\n\n    Alternatively, you may use keyword arguments to name a dependency and\n    provide the replacement value that the `MockDependencyProvider` should\n    inject.\n\n    Return the :attr:`MockDependencyProvider.dependency` for every dependency\n    specified in the (*dependencies) args so that calls to the replaced\n    dependencies can be inspected. Return a single object if only one\n    dependency was replaced, and a generator yielding the replacements in the\n    same order as ``dependencies`` otherwise.\n    Note that any replaced dependencies specified via kwargs `**dependency_map`\n    will not be returned.\n\n    Replacements are made on the container instance and have no effect on the\n    service class. New container instances are therefore unaffected by\n    replacements on previous instances.\n\n    **Usage**\n\n    ::\n\n        from nameko.rpc import RpcProxy, rpc\n        from nameko.standalone.rpc import ServiceRpcProxy\n\n        class ConversionService(object):\n            name = "conversions"\n\n            maths_rpc = RpcProxy("maths")\n\n            @rpc\n            def inches_to_cm(self, inches):\n                return self.maths_rpc.multiply(inches, 2.54)\n\n            @rpc\n            def cm_to_inches(self, cms):\n                return self.maths_rpc.divide(cms, 2.54)\n\n        container = ServiceContainer(ConversionService, config)\n        mock_maths_rpc = replace_dependencies(container, "maths_rpc")\n        mock_maths_rpc.divide.return_value = 39.37\n\n        container.start()\n\n        with ServiceRpcProxy(\'conversions\', config) as proxy:\n            proxy.cm_to_inches(100)\n\n        # assert that the dependency was called as expected\n        mock_maths_rpc.divide.assert_called_once_with(100, 2.54)\n\n\n    Providing a specific replacement by keyword:\n\n    ::\n\n        class StubMaths(object):\n\n            def divide(self, val1, val2):\n                return val1 / val2\n\n        replace_dependencies(container, maths_rpc=StubMaths())\n\n        container.start()\n\n        with ServiceRpcProxy(\'conversions\', config) as proxy:\n            assert proxy.cm_to_inches(127) == 50.0\n\n    '
    if set(dependencies).intersection(dependency_map):
        raise RuntimeError('Cannot replace the same dependency via both args and kwargs.')
    arg_replacements = OrderedDict(((dep, MagicMock()) for dep in dependencies))
    dependency_map.update(arg_replacements)
    _replace_dependencies(container, **dependency_map)
    res = (replacement for replacement in arg_replacements.values())
    if len(arg_replacements) == 1:
        return next(res)
    return res

def restrict_entrypoints(container, *entrypoints):
    if False:
        while True:
            i = 10
    ' Restrict the entrypoints on ``container`` to those named in\n    ``entrypoints``.\n\n    This method must be called before the container is started.\n\n    **Usage**\n\n    The following service definition has two entrypoints:\n\n    .. code-block:: python\n\n        class Service(object):\n            name = "service"\n\n            @timer(interval=1)\n            def foo(self, arg):\n                pass\n\n            @rpc\n            def bar(self, arg)\n                pass\n\n            @rpc\n            def baz(self, arg):\n                pass\n\n        container = ServiceContainer(Service, config)\n\n    To disable the timer entrypoint on ``foo``, leaving just the RPC\n    entrypoints:\n\n    .. code-block:: python\n\n        restrict_entrypoints(container, "bar", "baz")\n\n    Note that it is not possible to identify multiple entrypoints on the same\n    method individually.\n\n    '
    if container.started:
        raise RuntimeError('You must restrict entrypoints before the container is started.')
    entrypoint_deps = list(container.entrypoints)
    entrypoint_names = {ext.method_name for ext in entrypoint_deps}
    missing = set(entrypoints) - entrypoint_names
    if missing:
        raise ExtensionNotFound("Entrypoint(s) '{}' not found on {}.".format(missing, container))
    for entrypoint in entrypoint_deps:
        if entrypoint.method_name not in entrypoints:
            container.entrypoints.remove(entrypoint)

class Once(Entrypoint):
    """ Entrypoint that spawns a worker exactly once, as soon as
    the service container started.
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        expected_exceptions = kwargs.pop('expected_exceptions', ())
        sensitive_arguments = kwargs.pop('sensitive_arguments', ())
        sensitive_variables = kwargs.pop('sensitive_variables', ())
        self.args = args
        self.kwargs = kwargs
        super(Once, self).__init__(expected_exceptions=expected_exceptions, sensitive_arguments=sensitive_arguments, sensitive_variables=sensitive_variables)

    def start(self):
        if False:
            print('Hello World!')
        self.container.spawn_worker(self, self.args, self.kwargs)
once = Once.decorator
dummy = Entrypoint.decorator