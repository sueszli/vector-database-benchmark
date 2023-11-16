"""
A **fixture** provides a concept to simplify test support functionality
that needs a setup/cleanup cycle per scenario, feature or test-run.
A fixture is provided as fixture-function that contains the setup part and
cleanup part similar to :func:`contextlib.contextmanager` or `pytest.fixture`_.

.. _pytest.fixture: https://docs.pytest.org/en/latest/fixture.html

A fixture is used when:

* the (registered) fixture tag is used for a scenario or feature
* the :func:`.use_fixture()` is called in the environment file (normally)

.. sourcecode:: python

    # -- FILE: behave4my_project/fixtures.py (or: features/environment.py)
    from behave import fixture
    from somewhere.browser.firefox import FirefoxBrowser

    @fixture
    def browser_firefox(context, timeout=30, **kwargs):
        # -- SETUP-FIXTURE PART:
        context.browser = FirefoxBrowser(timeout, *args, **kwargs)
        yield context.browser
        # -- CLEANUP-FIXTURE PART:
        context.browser.shutdown()

.. sourcecode:: gherkin

    # -- FILE: features/use_fixture.feature
    Feature: Use Fixture in Scenario

        @fixture.browser.firefox
        Scenario: Use browser=firefox
          Given I use the browser
          ...
        # -- AFTER-SCENEARIO: Cleanup fixture.browser.firefox

.. sourcecode:: python

    # -- FILE: features/environment.py
    from behave import use_fixture
    from behave4my_project.fixtures import browser_firefox

    def before_tag(context, tag):
        if tag == "fixture.browser.firefox":
            # -- Performs fixture setup and registers fixture cleanup
            use_fixture(browser_firefox, context, timeout=10)

.. hidden:

    BEHAVIORAL DECISIONS:

    * Should scenario/feature be executed when fixture-setup fails
      (similar to before-hook failures) ?
      NO, scope is skipped, but after-hooks and cleanups are executed.

    * Should remaining fixture-setups be performed after first fixture fails?
      NO, first setup-error aborts the setup and execution of the scope.

    * Should remaining fixture-cleanups be performed when first cleanup-error
      occurs?
      YES, try to perform all fixture-cleanups and then reraise the
      first cleanup-error.


    OPEN ISSUES:

    * AUTO_CALL_REGISTERED_FIXTURE (planned in future):
        Run fixture setup before or after before-hooks?

    IDEAS:

    * Fixture registers itself in fixture registry (runtime context).
    * Code in before_tag() will either be replaced w/ fixture processing function
      or will be automatically be executed (AUTO_CALL_REGISTERED_FIXTURE)
    * Support fixture tags w/ parameters that are automatically parsed and
      passed to fixture function, like:
      @fixture(name="foo", pattern="{name}={browser}")
"""
import inspect

def iscoroutinefunction(func):
    if False:
        print('Hello World!')
    'Checks if a function is a coroutine-function, like:\n\n     * ``async def f(): ...`` (since Python 3.5)\n     * ``@asyncio.coroutine def f(): ...`` (since Python3)\n\n    .. note:: Compatibility helper\n\n        Avoids to import :mod:`asyncio` module directly (since Python3),\n        which in turns initializes the :mod:`logging` module as side-effect.\n\n    :param func:  Function to check.\n    :return: True, if function is a coroutine function.\n             False, otherwise.\n    '
    return getattr(func, '_is_coroutine', False) or (hasattr(inspect, 'iscoroutinefunction') and inspect.iscoroutinefunction(func))

def is_context_manager(func):
    if False:
        i = 10
        return i + 15
    'Checks if a fixture function provides context-manager functionality,\n    similar to :func`contextlib.contextmanager()` function decorator.\n\n    .. code-block:: python\n\n        @fixture\n        def foo(context, *args, **kwargs):\n            context.foo = setup_foo()\n            yield context.foo\n            cleanup_foo()\n\n        @fixture\n        def bar(context, *args, **kwargs):\n            context.bar = setup_bar()\n            return context.bar\n\n        assert is_context_manager(foo) is True      # Generator-function\n        assert is_context_manager(bar) is False     # Normal function\n\n    :param func:    Function to check.\n    :return: True, if function is a generator/context-manager function.\n             False, otherwise.\n    '
    genfunc = inspect.isgeneratorfunction(func)
    return genfunc and (not iscoroutinefunction(func))

class InvalidFixtureError(RuntimeError):
    """Raised when a fixture is invalid.
    This occurs when a generator-function with more than one yield statement
    is used as fixture-function.
    """

def _setup_fixture(fixture_func, context, *fixture_args, **fixture_kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Provides core functionality to setup a fixture and registers its\n    cleanup part (if needed).\n    '
    if is_context_manager(fixture_func):

        def cleanup_fixture():
            if False:
                i = 10
                return i + 15
            try:
                next(func_it)
            except StopIteration:
                return False
            else:
                message = 'Has more than one yield: %r' % fixture_func
                raise InvalidFixtureError(message)
        func_it = fixture_func(context, *fixture_args, **fixture_kwargs)
        context.add_cleanup(cleanup_fixture)
        setup_result = next(func_it)
    else:
        setup_result = fixture_func(context, *fixture_args, **fixture_kwargs)
    return setup_result

def use_fixture(fixture_func, context, *fixture_args, **fixture_kwargs):
    if False:
        i = 10
        return i + 15
    'Use fixture (function) and call it to perform its setup-part.\n\n    The fixture-function is similar to a :func:`contextlib.contextmanager`\n    (and contains a yield-statement to seperate setup and cleanup part).\n    If it contains a yield-statement, it registers a context-cleanup function\n    to the context object to perform the fixture-cleanup at the end of the\n    current scoped when the context layer is removed\n    (and all context-cleanup functions are called).\n\n    Therefore, fixture-cleanup is performed after scenario, feature or test-run\n    (depending when its fixture-setup is performed).\n\n    .. code-block:: python\n\n        # -- FILE: behave4my_project/fixtures.py (or: features/environment.py)\n        from behave import fixture\n        from somewhere.browser import FirefoxBrowser\n\n        @fixture(name="fixture.browser.firefox")\n        def browser_firefox(context, *args, **kwargs):\n            # -- SETUP-FIXTURE PART:\n            context.browser = FirefoxBrowser(*args, **kwargs)\n            yield context.browser\n            # -- CLEANUP-FIXTURE PART:\n            context.browser.shutdown()\n\n    .. code-block:: python\n\n        # -- FILE: features/environment.py\n        from behave import use_fixture\n        from behave4my_project.fixtures import browser_firefox\n\n        def before_tag(context, tag):\n            if tag == "fixture.browser.firefox":\n                use_fixture(browser_firefox, context, timeout=10)\n\n\n    :param fixture_func: Fixture function to use.\n    :param context: Context object to use\n    :param fixture_kwargs: Positional args, passed to the fixture function.\n    :param fixture_kwargs: Additional kwargs, passed to the fixture function.\n    :return: Setup result object (may be None).\n    '
    return _setup_fixture(fixture_func, context, *fixture_args, **fixture_kwargs)

def use_fixture_by_tag(tag, context, fixture_registry):
    if False:
        for i in range(10):
            print('nop')
    'Process any fixture-tag to perform :func:`use_fixture()` for its fixture.\n    If the fixture-tag is known, the fixture data is retrieved from the\n    fixture registry.\n\n    .. code-block:: python\n\n        # -- FILE: features/environment.py\n        from behave.fixture import use_fixture_by_tag\n        from behave4my_project.fixtures import browser_firefox, browser_chrome\n\n        # -- SCHEMA 1: fixture_func\n        fixture_registry1 = {\n            "fixture.browser.firefox": browser_firefox,\n            "fixture.browser.chrome":  browser_chrome,\n        }\n        # -- SCHEMA 2: fixture_func, fixture_args, fixture_kwargs\n        fixture_registry2 = {\n            "fixture.browser.firefox": (browser_firefox, (), dict(timeout=10)),\n            "fixture.browser.chrome":  (browser_chrome,  (), dict(timeout=12)),\n        }\n\n        def before_tag(context, tag):\n            if tag.startswith("fixture."):\n                return use_fixture_by_tag(tag, context, fixture_registry1):\n            # -- MORE: Tag processing steps ...\n\n\n    :param tag:     Fixture tag to process.\n    :param context: Runtime context object, used for :func:`use_fixture()`.\n    :param fixture_registry:  Registry maps fixture-tag to fixture data.\n    :return: Fixture-setup result (same as: use_fixture())\n    :raises LookupError: If fixture-tag/fixture is unknown.\n    :raises ValueError: If fixture data type is not supported.\n    '
    fixture_data = fixture_registry.get(tag, None)
    if fixture_data is None:
        raise LookupError('Unknown fixture-tag: %s' % tag)
    if callable(fixture_data):
        fixture_func = fixture_data
        return use_fixture(fixture_func, context)
    elif isinstance(fixture_data, (tuple, list)):
        assert len(fixture_data) == 3
        (fixture_func, fixture_args, fixture_kwargs) = fixture_data
        return use_fixture(fixture_func, context, *fixture_args, **fixture_kwargs)
    else:
        message = 'fixture_data: Expected tuple or fixture-func, but is: %r'
        raise ValueError(message % fixture_data)

def fixture_call_params(fixture_func, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return (fixture_func, args, kwargs)

def use_composite_fixture_with(context, fixture_funcs_with_params):
    if False:
        i = 10
        return i + 15
    'Helper function when complex fixtures should be created and\n    safe-cleanup is needed even if an setup-fixture-error occurs.\n\n    This function ensures that fixture-cleanup is performed\n    for every fixture that was setup before the setup-error occured.\n\n    .. code-block:: python\n\n        # -- BAD-EXAMPLE: Simplistic composite-fixture\n        # NOTE: Created fixtures (fixture1) are not cleaned up.\n        @fixture\n        def foo_and_bad0(context, *args, **kwargs):\n            the_fixture1 = setup_fixture_foo(*args, **kwargs)\n            the_fixture2 = setup_fixture_bar_with_error("OOPS-HERE")\n            yield (the_fixture1, the_fixture2)  # NOT_REACHED.\n            # -- NOT_REACHED: Due to fixture2-setup-error.\n            the_fixture1.cleanup()  # NOT-CALLED (SAD).\n            the_fixture2.cleanup()  # OOPS, the_fixture2 is None (if called).\n\n    .. code-block:: python\n\n        # -- GOOD-EXAMPLE: Sane composite-fixture\n        # NOTE: Fixture foo.cleanup() occurs even after fixture2-setup-error.\n        @fixture\n        def foo(context, *args, **kwargs):\n            the_fixture = setup_fixture_foo(*args, **kwargs)\n            yield the_fixture\n            cleanup_fixture_foo(the_fixture)\n\n        @fixture\n        def bad_with_setup_error(context, *args, **kwargs):\n            raise RuntimeError("BAD-FIXTURE-SETUP")\n\n        # -- SOLUTION 1: With use_fixture()\n        @fixture\n        def foo_and_bad1(context, *args, **kwargs):\n            the_fixture1 = use_fixture(foo, context, *args, **kwargs)\n            the_fixture2 = use_fixture(bad_with_setup_error, context, "OOPS")\n            return (the_fixture1, the_fixture2) # NOT_REACHED\n\n        # -- SOLUTION 2: With use_composite_fixture_with()\n        @fixture\n        def foo_and_bad2(context, *args, **kwargs):\n            the_fixture = use_composite_fixture_with(context, [\n                fixture_call_params(foo, *args, **kwargs),\n                fixture_call_params(bad_with_setup_error, "OOPS")\n             ])\n            return the_fixture\n\n    :param context:     Runtime context object, used for all fixtures.\n    :param fixture_funcs_with_params: List of fixture functions with params.\n    :return: List of created fixture objects.\n    '
    composite_fixture = []
    for (fixture_func, args, kwargs) in fixture_funcs_with_params:
        the_fixture = use_fixture(fixture_func, context, *args, **kwargs)
        composite_fixture.append(the_fixture)
    return composite_fixture

def fixture(func=None, name=None, pattern=None):
    if False:
        for i in range(10):
            print('nop')
    'Fixture decorator (currently mostly syntactic sugar).\n\n    .. code-block:: python\n\n        # -- FILE: features/environment.py\n        # CASE FIXTURE-GENERATOR-FUNCTION (like @contextlib.contextmanager):\n        @fixture\n        def foo(context, *args, **kwargs):\n            the_fixture = setup_fixture_foo(*args, **kwargs)\n            context.foo = the_fixture\n            yield the_fixture\n            cleanup_fixture_foo(the_fixture)\n\n        # CASE FIXTURE-FUNCTION: No cleanup or cleanup via context-cleanup.\n        @fixture(name="fixture.bar")\n        def bar(context, *args, **kwargs):\n            the_fixture = setup_fixture_bar(*args, **kwargs)\n            context.bar = the_fixture\n            context.add_cleanup(cleanup_fixture_bar, the_fixture.cleanup)\n            return the_fixture\n\n    :param name:    Specifies the fixture tag name (as string).\n\n    .. seealso::\n\n        * :func:`contextlib.contextmanager` decorator\n        * `@pytest.fixture`_\n    '

    def mark_as_fixture(func, name=None, pattern=None):
        if False:
            i = 10
            return i + 15
        func.name = name
        func.pattern = pattern
        func.behave_fixture = True
        return func
    if func is None:

        def decorator(func):
            if False:
                print('Hello World!')
            return mark_as_fixture(func, name, pattern=pattern)
        return decorator
    elif callable(func):
        return mark_as_fixture(func, name, pattern=pattern)
    else:
        message = 'Invalid func: func=%r, name=%r' % (func, name)
        raise TypeError(message)