from pyramid.scripting import prepare
from pyramid.scripts.common import get_config_loader

def setup_logging(config_uri, global_conf=None):
    if False:
        i = 10
        return i + 15
    '\n    Set up Python logging with the filename specified via ``config_uri``\n    (a string in the form ``filename#sectionname``).\n\n    Extra defaults can optionally be specified as a dict in ``global_conf``.\n    '
    loader = get_config_loader(config_uri)
    loader.setup_logging(global_conf)

def get_app(config_uri, name=None, options=None):
    if False:
        return 10
    'Return the WSGI application named ``name`` in the PasteDeploy\n    config file specified by ``config_uri``.\n\n    ``options``, if passed, should be a dictionary used as variable assignments\n    like ``{\'http_port\': 8080}``.  This is useful if e.g. ``%(http_port)s`` is\n    used in the config file.\n\n    If the ``name`` is None, this will attempt to parse the name from\n    the ``config_uri`` string expecting the format ``inifile#name``.\n    If no name is found, the name will default to "main".\n\n    '
    loader = get_config_loader(config_uri)
    return loader.get_wsgi_app(name, options)

def get_appsettings(config_uri, name=None, options=None):
    if False:
        while True:
            i = 10
    'Return a dictionary representing the key/value pairs in an ``app``\n    section within the file represented by ``config_uri``.\n\n    ``options``, if passed, should be a dictionary used as variable assignments\n    like ``{\'http_port\': 8080}``.  This is useful if e.g. ``%(http_port)s`` is\n    used in the config file.\n\n    If the ``name`` is None, this will attempt to parse the name from\n    the ``config_uri`` string expecting the format ``inifile#name``.\n    If no name is found, the name will default to "main".\n\n    '
    loader = get_config_loader(config_uri)
    return loader.get_wsgi_app_settings(name, options)

def bootstrap(config_uri, request=None, options=None):
    if False:
        print('Hello World!')
    "Load a WSGI application from the PasteDeploy config file specified\n    by ``config_uri``. The environment will be configured as if it is\n    currently serving ``request``, leaving a natural environment in place\n    to write scripts that can generate URLs and utilize renderers.\n\n    This function returns a dictionary with ``app``, ``root``, ``closer``,\n    ``request``, and ``registry`` keys.  ``app`` is the WSGI app loaded\n    (based on the ``config_uri``), ``root`` is the traversal root resource\n    of the Pyramid application, and ``closer`` is a parameterless callback\n    that may be called when your script is complete (it pops a threadlocal\n    stack).\n\n    .. note::\n\n       Most operations within :app:`Pyramid` expect to be invoked within the\n       context of a WSGI request, thus it's important when loading your\n       application to anchor it when executing scripts and other code that is\n       not normally invoked during active WSGI requests.\n\n    .. note::\n\n       For a complex config file containing multiple :app:`Pyramid`\n       applications, this function will setup the environment under the context\n       of the last-loaded :app:`Pyramid` application. You may load a specific\n       application yourself by using the lower-level functions\n       :meth:`pyramid.paster.get_app` and :meth:`pyramid.scripting.prepare` in\n       conjunction with :attr:`pyramid.config.global_registries`.\n\n    ``config_uri`` -- specifies the PasteDeploy config file to use for the\n    interactive shell. The format is ``inifile#name``. If the name is left\n    off, ``main`` will be assumed.\n\n    ``request`` -- specified to anchor the script to a given set of WSGI\n    parameters. For example, most people would want to specify the host,\n    scheme and port such that their script will generate URLs in relation\n    to those parameters. A request with default parameters is constructed\n    for you if none is provided. You can mutate the request's ``environ``\n    later to setup a specific host/port/scheme/etc.\n\n    ``options`` Is passed to get_app for use as variable assignments like\n    {'http_port': 8080} and then use %(http_port)s in the\n    config file.\n\n    This function may be used as a context manager to call the ``closer``\n    automatically:\n\n    .. code-block:: python\n\n       with bootstrap('development.ini') as env:\n           request = env['request']\n           # ...\n\n    See :ref:`writing_a_script` for more information about how to use this\n    function.\n\n    .. versionchanged:: 1.8\n\n       Added the ability to use the return value as a context manager.\n\n    .. versionchanged:: 2.0\n\n       Request finished callbacks added via\n       :meth:`pyramid.request.Request.add_finished_callback` will be invoked\n       by the ``closer``.\n\n    "
    app = get_app(config_uri, options=options)
    env = prepare(request)
    env['app'] = app
    return env