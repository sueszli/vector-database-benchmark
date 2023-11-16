"""
Contains base Spawner class & default implementation
"""
import ast
import json
import os
import shlex
import shutil
import signal
import sys
import warnings
from inspect import signature
from subprocess import Popen
from tempfile import mkdtemp
from textwrap import dedent
from urllib.parse import urlparse
from async_generator import aclosing
from sqlalchemy import inspect
from tornado.ioloop import PeriodicCallback
from traitlets import Any, Bool, Dict, Float, Instance, Integer, List, Unicode, Union, default, observe, validate
from traitlets.config import LoggingConfigurable
from . import orm
from .objects import Server
from .roles import roles_to_scopes
from .traitlets import ByteSpecification, Callable, Command
from .utils import AnyTimeoutError, exponential_backoff, maybe_future, random_port, url_escape_path, url_path_join
if os.name == 'nt':
    import psutil

def _quote_safe(s):
    if False:
        for i in range(10):
            print('nop')
    'pass a string that is safe on the command-line\n\n    traitlets may parse literals on the command-line, e.g. `--ip=123` will be the number 123 instead of the *string* 123.\n    wrap valid literals in repr to ensure they are safe\n    '
    try:
        val = ast.literal_eval(s)
    except Exception:
        return s
    else:
        return repr(s)

class Spawner(LoggingConfigurable):
    """Base class for spawning single-user notebook servers.

    Subclass this, and override the following methods:

    - load_state
    - get_state
    - start
    - stop
    - poll

    As JupyterHub supports multiple users, an instance of the Spawner subclass
    is created for each user. If there are 20 JupyterHub users, there will be 20
    instances of the subclass.
    """
    _spawn_pending = False
    _start_pending = False
    _stop_pending = False
    _proxy_pending = False
    _check_pending = False
    _waiting_for_response = False
    _jupyterhub_version = None
    _spawn_future = None

    @property
    def _log_name(self):
        if False:
            while True:
                i = 10
        'Return username:servername or username\n\n        Used in logging for consistency with named servers.\n        '
        if self.user:
            user_name = self.user.name
        else:
            user_name = '(no user)'
        if self.name:
            return f'{user_name}:{self.name}'
        else:
            return user_name

    @property
    def _failed(self):
        if False:
            return 10
        'Did the last spawn fail?'
        return not self.active and self._spawn_future and self._spawn_future.done() and self._spawn_future.exception()

    @property
    def pending(self):
        if False:
            return 10
        'Return the current pending event, if any\n\n        Return False if nothing is pending.\n        '
        if self._spawn_pending:
            return 'spawn'
        elif self._stop_pending:
            return 'stop'
        elif self._check_pending:
            return 'check'
        return None

    @property
    def ready(self):
        if False:
            i = 10
            return i + 15
        'Is this server ready to use?\n\n        A server is not ready if an event is pending.\n        '
        if self.pending:
            return False
        if self.server is None:
            return False
        return True

    @property
    def active(self):
        if False:
            for i in range(10):
                print('nop')
        'Return True if the server is active.\n\n        This includes fully running and ready or any pending start/stop event.\n        '
        return bool(self.pending or self.ready)
    authenticator = Any()
    hub = Any()
    orm_spawner = Any()
    cookie_options = Dict()
    db = Any()

    @default('db')
    def _deprecated_db(self):
        if False:
            print('Hello World!')
        self.log.warning(dedent('\n                The shared database session at Spawner.db is deprecated, and will be removed.\n                Please manage your own database and connections.\n\n                Contact JupyterHub at https://github.com/jupyterhub/jupyterhub/issues/3700\n                if you have questions or ideas about direct database needs for your Spawner.\n                '))
        return self._deprecated_db_session
    _deprecated_db_session = Any()

    @observe('orm_spawner')
    def _orm_spawner_changed(self, change):
        if False:
            while True:
                i = 10
        if change.new and change.new.server:
            self._server = Server(orm_server=change.new.server)
        else:
            self._server = None
    user = Any()

    def __init_subclass__(cls, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init_subclass__()
        missing = []
        for attr in ('start', 'stop', 'poll'):
            if getattr(Spawner, attr) is getattr(cls, attr):
                missing.append(attr)
        if missing:
            raise NotImplementedError('class `{}` needs to redefine the `start`,`stop` and `poll` methods. `{}` not redefined.'.format(cls.__name__, '`, `'.join(missing)))
    proxy_spec = Unicode()

    @property
    def last_activity(self):
        if False:
            while True:
                i = 10
        return self.orm_spawner.last_activity
    _server = None

    @property
    def server(self):
        if False:
            i = 10
            return i + 15
        if not self.orm_spawner:
            return self._server
        orm_server = self.orm_spawner.server
        if orm_server is not None and (self._server is None or orm_server is not self._server.orm_server):
            self._server = Server(orm_server=self.orm_spawner.server)
        elif orm_server is None:
            self._server = None
        return self._server

    @server.setter
    def server(self, server):
        if False:
            print('Hello World!')
        self._server = server
        if self.orm_spawner is not None:
            if server is not None and server.orm_server == self.orm_spawner.server:
                return
            if self.orm_spawner.server is not None:
                db = inspect(self.orm_spawner.server).session
                db.delete(self.orm_spawner.server)
            if server is None:
                self.orm_spawner.server = None
            else:
                if server.orm_server is None:
                    self.log.warning(f'No ORM server for {self._log_name}')
                self.orm_spawner.server = server.orm_server
        elif server is not None:
            self.log.warning(f'Setting Spawner.server for {self._log_name} with no underlying orm_spawner')

    @property
    def name(self):
        if False:
            while True:
                i = 10
        if self.orm_spawner:
            return self.orm_spawner.name
        return ''
    internal_ssl = Bool(False)
    internal_trust_bundles = Dict()
    internal_certs_location = Unicode('')
    cert_paths = Dict()
    admin_access = Bool(False)
    api_token = Unicode()
    oauth_client_id = Unicode()

    @property
    def oauth_scopes(self):
        if False:
            print('Hello World!')
        warnings.warn('Spawner.oauth_scopes is deprecated in JupyterHub 2.3.\n\n            Use Spawner.oauth_access_scopes\n            ', DeprecationWarning, stacklevel=2)
        return self.oauth_access_scopes
    oauth_access_scopes = List(Unicode(), help='The scope(s) needed to access this server')

    @default('oauth_access_scopes')
    def _default_access_scopes(self):
        if False:
            i = 10
            return i + 15
        return [f'access:servers!server={self.user.name}/{self.name}', f'access:servers!user={self.user.name}']
    handler = Any()
    oauth_roles = Union([Callable(), List()], help="Allowed roles for oauth tokens.\n\n        Deprecated in 3.0: use oauth_client_allowed_scopes\n\n        This sets the maximum and default roles\n        assigned to oauth tokens issued by a single-user server's\n        oauth client (i.e. tokens stored in browsers after authenticating with the server),\n        defining what actions the server can take on behalf of logged-in users.\n\n        Default is an empty list, meaning minimal permissions to identify users,\n        no actions can be taken on their behalf.\n        ").tag(config=True)
    oauth_client_allowed_scopes = Union([Callable(), List()], help="Allowed scopes for oauth tokens issued by this server's oauth client.\n\n        This sets the maximum and default scopes\n        assigned to oauth tokens issued by a single-user server's\n        oauth client (i.e. tokens stored in browsers after authenticating with the server),\n        defining what actions the server can take on behalf of logged-in users.\n\n        Default is an empty list, meaning minimal permissions to identify users,\n        no actions can be taken on their behalf.\n\n        If callable, will be called with the Spawner as a single argument.\n        Callables may be async.\n    ").tag(config=True)

    async def _get_oauth_client_allowed_scopes(self):
        """Private method: get oauth allowed scopes

        Handle:

        - oauth_client_allowed_scopes
        - callable config
        - deprecated oauth_roles config
        - access_scopes
        """
        scopes = []
        if self.oauth_client_allowed_scopes:
            allowed_scopes = self.oauth_client_allowed_scopes
            if callable(allowed_scopes):
                allowed_scopes = allowed_scopes(self)
                if inspect.isawaitable(allowed_scopes):
                    allowed_scopes = await allowed_scopes
            scopes.extend(allowed_scopes)
        if self.oauth_roles:
            if scopes:
                warnings.warn(f'Ignoring deprecated Spawner.oauth_roles={self.oauth_roles} in favor of Spawner.oauth_client_allowed_scopes.')
            else:
                role_names = self.oauth_roles
                if callable(role_names):
                    role_names = role_names(self)
                roles = list(self.db.query(orm.Role).filter(orm.Role.name.in_(role_names)))
                if len(roles) != len(role_names):
                    missing_roles = set(role_names).difference({role.name for role in roles})
                    raise ValueError(f"No such role(s): {', '.join(missing_roles)}")
                scopes.extend(roles_to_scopes(roles))
        scopes.append(f'access:servers!server={self.user.name}/{self.name}')
        return sorted(set(scopes))
    server_token_scopes = Union([List(Unicode()), Callable()], help='The list of scopes to request for $JUPYTERHUB_API_TOKEN\n\n        If not specified, the scopes in the `server` role will be used\n        (unchanged from pre-4.0).\n\n        If callable, will be called with the Spawner instance as its sole argument\n        (JupyterHub user available as spawner.user).\n\n        JUPYTERHUB_API_TOKEN will be assigned the _subset_ of these scopes\n        that are held by the user (as in oauth_client_allowed_scopes).\n\n        .. versionadded:: 4.0\n        ').tag(config=True)
    will_resume = Bool(False, help='Whether the Spawner will resume on next start\n\n\n        Default is False where each launch of the Spawner will be a new instance.\n        If True, an existing Spawner will resume instead of starting anew\n        (e.g. resuming a Docker container),\n        and API tokens in use when the Spawner stops will not be deleted.\n        ')
    ip = Unicode('127.0.0.1', help="\n        The IP address (or hostname) the single-user server should listen on.\n\n        Usually either '127.0.0.1' (default) or '0.0.0.0'.\n\n        The JupyterHub proxy implementation should be able to send packets to this interface.\n\n        Subclasses which launch remotely or in containers\n        should override the default to '0.0.0.0'.\n\n        .. versionchanged:: 2.0\n            Default changed to '127.0.0.1', from ''.\n            In most cases, this does not result in a change in behavior,\n            as '' was interpreted as 'unspecified',\n            which used the subprocesses' own default, itself usually '127.0.0.1'.\n        ").tag(config=True)
    port = Integer(0, help='\n        The port for single-user servers to listen on.\n\n        Defaults to `0`, which uses a randomly allocated port number each time.\n\n        If set to a non-zero value, all Spawners will use the same port,\n        which only makes sense if each server is on a different address,\n        e.g. in containers.\n\n        New in version 0.7.\n        ').tag(config=True)
    consecutive_failure_limit = Integer(0, help='\n        Maximum number of consecutive failures to allow before\n        shutting down JupyterHub.\n\n        This helps JupyterHub recover from a certain class of problem preventing launch\n        in contexts where the Hub is automatically restarted (e.g. systemd, docker, kubernetes).\n\n        A limit of 0 means no limit and consecutive failures will not be tracked.\n        ').tag(config=True)
    start_timeout = Integer(60, help='\n        Timeout (in seconds) before giving up on starting of single-user server.\n\n        This is the timeout for start to return, not the timeout for the server to respond.\n        Callers of spawner.start will assume that startup has failed if it takes longer than this.\n        start should return when the server process is started and its location is known.\n        ').tag(config=True)
    http_timeout = Integer(30, help='\n        Timeout (in seconds) before giving up on a spawned HTTP server\n\n        Once a server has successfully been spawned, this is the amount of time\n        we wait before assuming that the server is unable to accept\n        connections.\n        ').tag(config=True)
    poll_interval = Integer(30, help="\n        Interval (in seconds) on which to poll the spawner for single-user server's status.\n\n        At every poll interval, each spawner's `.poll` method is called, which checks\n        if the single-user server is still running. If it isn't running, then JupyterHub modifies\n        its own state accordingly and removes appropriate routes from the configurable proxy.\n        ").tag(config=True)
    _callbacks = List()
    _poll_callback = Any()
    debug = Bool(False, help='Enable debug-logging of the single-user server').tag(config=True)
    options_form = Union([Unicode(), Callable()], help='\n        An HTML form for options a user can specify on launching their server.\n\n        The surrounding `<form>` element and the submit button are already provided.\n\n        For example:\n\n        .. code:: html\n\n            Set your key:\n            <input name="key" val="default_key"></input>\n            <br>\n            Choose a letter:\n            <select name="letter" multiple="true">\n              <option value="A">The letter A</option>\n              <option value="B">The letter B</option>\n            </select>\n\n        The data from this form submission will be passed on to your spawner in `self.user_options`\n\n        Instead of a form snippet string, this could also be a callable that takes as one\n        parameter the current spawner instance and returns a string. The callable will\n        be called asynchronously if it returns a future, rather than a str. Note that\n        the interface of the spawner class is not deemed stable across versions,\n        so using this functionality might cause your JupyterHub upgrades to break.\n    ').tag(config=True)

    async def get_options_form(self):
        """Get the options form

        Returns:
          Future (str): the content of the options form presented to the user
          prior to starting a Spawner.

        .. versionadded:: 0.9
        """
        if callable(self.options_form):
            options_form = await maybe_future(self.options_form(self))
        else:
            options_form = self.options_form
        return options_form
    options_from_form = Callable(help='\n        Interpret HTTP form data\n\n        Form data will always arrive as a dict of lists of strings.\n        Override this function to understand single-values, numbers, etc.\n\n        This should coerce form data into the structure expected by self.user_options,\n        which must be a dict, and should be JSON-serializeable,\n        though it can contain bytes in addition to standard JSON data types.\n\n        This method should not have any side effects.\n        Any handling of `user_options` should be done in `.start()`\n        to ensure consistent behavior across servers\n        spawned via the API and form submission page.\n\n        Instances will receive this data on self.user_options, after passing through this function,\n        prior to `Spawner.start`.\n\n        .. versionchanged:: 1.0\n            user_options are persisted in the JupyterHub database to be reused\n            on subsequent spawns if no options are given.\n            user_options is serialized to JSON as part of this persistence\n            (with additional support for bytes in case of uploaded file data),\n            and any non-bytes non-jsonable values will be replaced with None\n            if the user_options are re-used.\n        ').tag(config=True)

    @default('options_from_form')
    def _options_from_form(self):
        if False:
            print('Hello World!')
        return self._default_options_from_form

    def _default_options_from_form(self, form_data):
        if False:
            print('Hello World!')
        return form_data

    def run_options_from_form(self, form_data):
        if False:
            i = 10
            return i + 15
        sig = signature(self.options_from_form)
        if 'spawner' in sig.parameters:
            return self.options_from_form(form_data, spawner=self)
        else:
            return self.options_from_form(form_data)

    def options_from_query(self, query_data):
        if False:
            for i in range(10):
                print('nop')
        'Interpret query arguments passed to /spawn\n\n        Query arguments will always arrive as a dict of unicode strings.\n        Override this function to understand single-values, numbers, etc.\n\n        By default, options_from_form is called from this function. You can however override\n        this function if you need to process the query arguments differently.\n\n        This should coerce form data into the structure expected by self.user_options,\n        which must be a dict, and should be JSON-serializeable,\n        though it can contain bytes in addition to standard JSON data types.\n\n        This method should not have any side effects.\n        Any handling of `user_options` should be done in `.start()`\n        to ensure consistent behavior across servers\n        spawned via the API and form submission page.\n\n        Instances will receive this data on self.user_options, after passing through this function,\n        prior to `Spawner.start`.\n\n        .. versionadded:: 1.2\n            user_options are persisted in the JupyterHub database to be reused\n            on subsequent spawns if no options are given.\n            user_options is serialized to JSON as part of this persistence\n            (with additional support for bytes in case of uploaded file data),\n            and any non-bytes non-jsonable values will be replaced with None\n            if the user_options are re-used.\n        '
        return self.options_from_form(query_data)
    user_options = Dict(help="\n        Dict of user specified options for the user's spawned instance of a single-user server.\n\n        These user options are usually provided by the `options_form` displayed to the user when they start\n        their server.\n        ")
    env_keep = List(['PATH', 'PYTHONPATH', 'CONDA_ROOT', 'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV', 'LANG', 'LC_ALL', 'JUPYTERHUB_SINGLEUSER_APP'], help="\n        List of environment variables for the single-user server to inherit from the JupyterHub process.\n\n        This list is used to ensure that sensitive information in the JupyterHub process's environment\n        (such as `CONFIGPROXY_AUTH_TOKEN`) is not passed to the single-user server's process.\n        ").tag(config=True)
    env = Dict(help='Deprecated: use Spawner.get_env or Spawner.environment\n\n    - extend Spawner.get_env for adding required env in Spawner subclasses\n    - Spawner.environment for config-specified env\n    ')
    environment = Dict(help="\n        Extra environment variables to set for the single-user server's process.\n\n        Environment variables that end up in the single-user server's process come from 3 sources:\n          - This `environment` configurable\n          - The JupyterHub process' environment variables that are listed in `env_keep`\n          - Variables to establish contact between the single-user notebook and the hub (such as JUPYTERHUB_API_TOKEN)\n\n        The `environment` configurable should be set by JupyterHub administrators to add\n        installation specific environment variables. It is a dict where the key is the name of the environment\n        variable, and the value can be a string or a callable. If it is a callable, it will be called\n        with one parameter (the spawner instance), and should return a string fairly quickly (no blocking\n        operations please!).\n\n        Note that the spawner class' interface is not guaranteed to be exactly same across upgrades,\n        so if you are using the callable take care to verify it continues to work after upgrades!\n\n        .. versionchanged:: 1.2\n            environment from this configuration has highest priority,\n            allowing override of 'default' env variables,\n            such as JUPYTERHUB_API_URL.\n        ").tag(config=True)
    cmd = Command(['jupyterhub-singleuser'], allow_none=True, help='\n        The command used for starting the single-user server.\n\n        Provide either a string or a list containing the path to the startup script command. Extra arguments,\n        other than this path, should be provided via `args`.\n\n        This is usually set if you want to start the single-user server in a different python\n        environment (with virtualenv/conda) than JupyterHub itself.\n\n        Some spawners allow shell-style expansion here, allowing you to use environment variables.\n        Most, including the default, do not. Consult the documentation for your spawner to verify!\n        ').tag(config=True)
    args = List(Unicode(), help='\n        Extra arguments to be passed to the single-user server.\n\n        Some spawners allow shell-style expansion here, allowing you to use environment variables here.\n        Most, including the default, do not. Consult the documentation for your spawner to verify!\n        ').tag(config=True)
    notebook_dir = Unicode(help="\n        Path to the notebook directory for the single-user server.\n\n        The user sees a file listing of this directory when the notebook interface is started. The\n        current interface does not easily allow browsing beyond the subdirectories in this directory's\n        tree.\n\n        `~` will be expanded to the home directory of the user, and {username} will be replaced\n        with the name of the user.\n\n        Note that this does *not* prevent users from accessing files outside of this path! They\n        can do so with many other means.\n        ").tag(config=True)
    default_url = Unicode(help="\n        The URL the single-user server should start in.\n\n        `{username}` will be expanded to the user's username\n\n        Example uses:\n\n        - You can set `notebook_dir` to `/` and `default_url` to `/tree/home/{username}` to allow people to\n          navigate the whole filesystem from their notebook server, but still start in their home directory.\n        - Start with `/notebooks` instead of `/tree` if `default_url` points to a notebook instead of a directory.\n        - You can set this to `/lab` to have JupyterLab start by default, rather than Jupyter Notebook.\n        ").tag(config=True)

    @validate('notebook_dir', 'default_url')
    def _deprecate_percent_u(self, proposal):
        if False:
            i = 10
            return i + 15
        v = proposal['value']
        if '%U' in v:
            self.log.warning('%%U for username in %s is deprecated in JupyterHub 0.7, use {username}', proposal['trait'].name)
            v = v.replace('%U', '{username}')
            self.log.warning('Converting %r to %r', proposal['value'], v)
        return v
    disable_user_config = Bool(False, help="\n        Disable per-user configuration of single-user servers.\n\n        When starting the user's single-user server, any config file found in the user's $HOME directory\n        will be ignored.\n\n        Note: a user could circumvent this if the user modifies their Python environment, such as when\n        they have their own conda environments / virtualenvs / containers.\n        ").tag(config=True)
    mem_limit = ByteSpecification(None, help='\n        Maximum number of bytes a single-user notebook server is allowed to use.\n\n        Allows the following suffixes:\n          - K -> Kilobytes\n          - M -> Megabytes\n          - G -> Gigabytes\n          - T -> Terabytes\n\n        If the single user server tries to allocate more memory than this,\n        it will fail. There is no guarantee that the single-user notebook server\n        will be able to allocate this much memory - only that it can not\n        allocate more than this.\n\n        **This is a configuration setting. Your spawner must implement support\n        for the limit to work.** The default spawner, `LocalProcessSpawner`,\n        does **not** implement this support. A custom spawner **must** add\n        support for this setting for it to be enforced.\n        ').tag(config=True)
    cpu_limit = Float(None, allow_none=True, help='\n        Maximum number of cpu-cores a single-user notebook server is allowed to use.\n\n        If this value is set to 0.5, allows use of 50% of one CPU.\n        If this value is set to 2, allows use of up to 2 CPUs.\n\n        The single-user notebook server will never be scheduled by the kernel to\n        use more cpu-cores than this. There is no guarantee that it can\n        access this many cpu-cores.\n\n        **This is a configuration setting. Your spawner must implement support\n        for the limit to work.** The default spawner, `LocalProcessSpawner`,\n        does **not** implement this support. A custom spawner **must** add\n        support for this setting for it to be enforced.\n        ').tag(config=True)
    mem_guarantee = ByteSpecification(None, help='\n        Minimum number of bytes a single-user notebook server is guaranteed to have available.\n\n        Allows the following suffixes:\n          - K -> Kilobytes\n          - M -> Megabytes\n          - G -> Gigabytes\n          - T -> Terabytes\n\n        **This is a configuration setting. Your spawner must implement support\n        for the limit to work.** The default spawner, `LocalProcessSpawner`,\n        does **not** implement this support. A custom spawner **must** add\n        support for this setting for it to be enforced.\n        ').tag(config=True)
    cpu_guarantee = Float(None, allow_none=True, help='\n        Minimum number of cpu-cores a single-user notebook server is guaranteed to have available.\n\n        If this value is set to 0.5, allows use of 50% of one CPU.\n        If this value is set to 2, allows use of up to 2 CPUs.\n\n        **This is a configuration setting. Your spawner must implement support\n        for the limit to work.** The default spawner, `LocalProcessSpawner`,\n        does **not** implement this support. A custom spawner **must** add\n        support for this setting for it to be enforced.\n        ').tag(config=True)
    progress_ready_hook = Any(help='\n        An optional hook function that you can implement to modify the\n        ready event, which will be shown to the user on the spawn progress page when their server\n        is ready.\n\n        This can be set independent of any concrete spawner implementation.\n\n        This maybe a coroutine.\n\n        Example::\n\n            async def my_ready_hook(spawner, ready_event):\n                ready_event["html_message"] = f"Server {spawner.name} is ready for {spawner.user.name}"\n                return ready_event\n\n            c.Spawner.progress_ready_hook = my_ready_hook\n\n        ').tag(config=True)
    pre_spawn_hook = Any(help="\n        An optional hook function that you can implement to do some\n        bootstrapping work before the spawner starts. For example, create a\n        directory for your user or load initial content.\n\n        This can be set independent of any concrete spawner implementation.\n\n        This maybe a coroutine.\n\n        Example::\n\n            from subprocess import check_call\n            def my_hook(spawner):\n                username = spawner.user.name\n                check_call(['./examples/bootstrap-script/bootstrap.sh', username])\n\n            c.Spawner.pre_spawn_hook = my_hook\n\n        ").tag(config=True)
    post_stop_hook = Any(help='\n        An optional hook function that you can implement to do work after\n        the spawner stops.\n\n        This can be set independent of any concrete spawner implementation.\n        ').tag(config=True)
    auth_state_hook = Any(help='\n        An optional hook function that you can implement to pass `auth_state`\n        to the spawner after it has been initialized but before it starts.\n        The `auth_state` dictionary may be set by the `.authenticate()`\n        method of the authenticator.  This hook enables you to pass some\n        or all of that information to your spawner.\n\n        Example::\n\n            def userdata_hook(spawner, auth_state):\n                spawner.userdata = auth_state["userdata"]\n\n            c.Spawner.auth_state_hook = userdata_hook\n\n        ').tag(config=True)
    hub_connect_url = Unicode(None, allow_none=True, help="\n        The URL the single-user server should connect to the Hub.\n\n        If the Hub URL set in your JupyterHub config is not reachable\n        from spawned notebooks, you can set differnt URL by this config.\n\n        Is None if you don't need to change the URL.\n        ").tag(config=True)

    def load_state(self, state):
        if False:
            i = 10
            return i + 15
        "Restore state of spawner from database.\n\n        Called for each user's spawner after the hub process restarts.\n\n        `state` is a dict that'll contain the value returned by `get_state` of\n        the spawner, or {} if the spawner hasn't persisted any state yet.\n\n        Override in subclasses to restore any extra state that is needed to track\n        the single-user server for that user. Subclasses should call super().\n        "

    def get_state(self):
        if False:
            i = 10
            return i + 15
        'Save state of spawner into database.\n\n        A black box of extra state for custom spawners. The returned value of this is\n        passed to `load_state`.\n\n        Subclasses should call `super().get_state()`, augment the state returned from\n        there, and return that state.\n\n        Returns\n        -------\n        state: dict\n            a JSONable dict of state\n        '
        state = {}
        return state

    def clear_state(self):
        if False:
            for i in range(10):
                print('nop')
        'Clear any state that should be cleared when the single-user server stops.\n\n        State that should be preserved across single-user server instances should not be cleared.\n\n        Subclasses should call super, to ensure that state is properly cleared.\n        '
        self.api_token = ''

    def get_env(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the environment dict to use for the Spawner.\n\n        This applies things like `env_keep`, anything defined in `Spawner.environment`,\n        and adds the API token to the env.\n\n        When overriding in subclasses, subclasses must call `super().get_env()`, extend the\n        returned dict and return it.\n\n        Use this to access the env in Spawner.start to allow extension in subclasses.\n        '
        env = {}
        if self.env:
            warnings.warn('Spawner.env is deprecated, found %s' % self.env, DeprecationWarning)
            env.update(self.env)
        for key in self.env_keep:
            if key in os.environ:
                env[key] = os.environ[key]
        env['JUPYTERHUB_API_TOKEN'] = self.api_token
        env['JPY_API_TOKEN'] = self.api_token
        if self.admin_access:
            env['JUPYTERHUB_ADMIN_ACCESS'] = '1'
        env['JUPYTERHUB_CLIENT_ID'] = self.oauth_client_id
        if self.cookie_options:
            env['JUPYTERHUB_COOKIE_OPTIONS'] = json.dumps(self.cookie_options)
        env['JUPYTERHUB_HOST'] = self.hub.public_host
        env['JUPYTERHUB_OAUTH_CALLBACK_URL'] = url_path_join(self.user.url, url_escape_path(self.name), 'oauth_callback')
        env['JUPYTERHUB_OAUTH_SCOPES'] = json.dumps(self.oauth_access_scopes)
        env['JUPYTERHUB_OAUTH_ACCESS_SCOPES'] = json.dumps(self.oauth_access_scopes)
        env['JUPYTERHUB_OAUTH_CLIENT_ALLOWED_SCOPES'] = json.dumps(self.oauth_client_allowed_scopes)
        env['JUPYTERHUB_USER'] = self.user.name
        env['JUPYTERHUB_SERVER_NAME'] = self.name
        if self.hub_connect_url is not None:
            hub_api_url = url_path_join(self.hub_connect_url, urlparse(self.hub.api_url).path)
        else:
            hub_api_url = self.hub.api_url
        env['JUPYTERHUB_API_URL'] = hub_api_url
        env['JUPYTERHUB_ACTIVITY_URL'] = url_path_join(hub_api_url, 'users', getattr(self.user, 'escaped_name', self.user.name), 'activity')
        env['JUPYTERHUB_BASE_URL'] = self.hub.base_url[:-4]
        if self.server:
            base_url = self.server.base_url
            env['JUPYTERHUB_SERVICE_PREFIX'] = self.server.base_url
        else:
            base_url = '/'
        proto = 'https' if self.internal_ssl else 'http'
        bind_url = f'{proto}://{self.ip}:{self.port}{base_url}'
        env['JUPYTERHUB_SERVICE_URL'] = bind_url
        if self.mem_limit:
            env['MEM_LIMIT'] = str(self.mem_limit)
        if self.mem_guarantee:
            env['MEM_GUARANTEE'] = str(self.mem_guarantee)
        if self.cpu_limit:
            env['CPU_LIMIT'] = str(self.cpu_limit)
        if self.cpu_guarantee:
            env['CPU_GUARANTEE'] = str(self.cpu_guarantee)
        if self.cert_paths:
            env['JUPYTERHUB_SSL_KEYFILE'] = self.cert_paths['keyfile']
            env['JUPYTERHUB_SSL_CERTFILE'] = self.cert_paths['certfile']
            env['JUPYTERHUB_SSL_CLIENT_CA'] = self.cert_paths['cafile']
        if self.notebook_dir:
            notebook_dir = self.format_string(self.notebook_dir)
            env['JUPYTERHUB_ROOT_DIR'] = notebook_dir
        if self.default_url:
            default_url = self.format_string(self.default_url)
            env['JUPYTERHUB_DEFAULT_URL'] = default_url
        if self.debug:
            env['JUPYTERHUB_DEBUG'] = '1'
        if self.disable_user_config:
            env['JUPYTERHUB_DISABLE_USER_CONFIG'] = '1'
        for (key, value) in self.environment.items():
            if callable(value):
                env[key] = value(self)
            else:
                env[key] = value
        return env

    async def get_url(self):
        """Get the URL to connect to the server

        Sometimes JupyterHub may ask the Spawner for its url.
        This can occur e.g. when JupyterHub has restarted while a server was not finished starting,
        giving Spawners a chance to recover the URL where their server is running.

        The default is to trust that JupyterHub has the right information.
        Only override this method in Spawners that know how to
        check the correct URL for the servers they start.

        This will only be asked of Spawners that claim to be running
        (`poll()` returns `None`).
        """
        return self.server.url

    def template_namespace(self):
        if False:
            return 10
        "Return the template namespace for format-string formatting.\n\n        Currently used on default_url and notebook_dir.\n\n        Subclasses may add items to the available namespace.\n\n        The default implementation includes::\n\n            {\n              'username': user.name,\n              'base_url': users_base_url,\n            }\n\n        Returns:\n\n            ns (dict): namespace for string formatting.\n        "
        d = {'username': self.user.name}
        if self.server:
            d['base_url'] = self.server.base_url
        return d

    def format_string(self, s):
        if False:
            while True:
                i = 10
        'Render a Python format string\n\n        Uses :meth:`Spawner.template_namespace` to populate format namespace.\n\n        Args:\n\n            s (str): Python format-string to be formatted.\n\n        Returns:\n\n            str: Formatted string, rendered\n        '
        return s.format(**self.template_namespace())
    trusted_alt_names = List(Unicode())
    ssl_alt_names = List(Unicode(), config=True, help='List of SSL alt names\n\n        May be set in config if all spawners should have the same value(s),\n        or set at runtime by Spawner that know their names.\n        ')

    @default('ssl_alt_names')
    def _default_ssl_alt_names(self):
        if False:
            return 10
        return list(self.trusted_alt_names)
    ssl_alt_names_include_local = Bool(True, config=True, help='Whether to include `DNS:localhost`, `IP:127.0.0.1` in alt names')

    async def create_certs(self):
        """Create and set ownership for the certs to be used for internal ssl

        Keyword Arguments:
            alt_names (list): a list of alternative names to identify the
            server by, see:
            https://en.wikipedia.org/wiki/Subject_Alternative_Name

            override: override the default_names with the provided alt_names

        Returns:
            dict: Path to cert files and CA

        This method creates certs for use with the singleuser notebook. It
        enables SSL and ensures that the notebook can perform bi-directional
        SSL auth with the hub (verification based on CA).

        If the singleuser host has a name or ip other than localhost,
        an appropriate alternative name(s) must be passed for ssl verification
        by the hub to work. For example, for Jupyter hosts with an IP of
        10.10.10.10 or DNS name of jupyter.example.com, this would be:

        alt_names=["IP:10.10.10.10"]
        alt_names=["DNS:jupyter.example.com"]

        respectively. The list can contain both the IP and DNS names to refer
        to the host by either IP or DNS name (note the `default_names` below).
        """
        from certipy import Certipy
        default_names = ['DNS:localhost', 'IP:127.0.0.1']
        alt_names = []
        alt_names.extend(self.ssl_alt_names)
        if self.ssl_alt_names_include_local:
            alt_names = default_names + alt_names
        self.log.info('Creating certs for %s: %s', self._log_name, ';'.join(alt_names))
        common_name = self.user.name or 'service'
        certipy = Certipy(store_dir=self.internal_certs_location)
        notebook_component = 'notebooks-ca'
        notebook_key_pair = certipy.create_signed_pair('user-' + common_name, notebook_component, alt_names=alt_names, overwrite=True)
        paths = {'keyfile': notebook_key_pair['files']['key'], 'certfile': notebook_key_pair['files']['cert'], 'cafile': self.internal_trust_bundles[notebook_component]}
        return paths

    async def move_certs(self, paths):
        """Takes certificate paths and makes them available to the notebook server

        Arguments:
            paths (dict): a list of paths for key, cert, and CA.
                These paths will be resolvable and readable by the Hub process,
                but not necessarily by the notebook server.

        Returns:
            dict: a list (potentially altered) of paths for key, cert, and CA.
                These paths should be resolvable and readable by the notebook
                server to be launched.


        `.move_certs` is called after certs for the singleuser notebook have
        been created by create_certs.

        By default, certs are created in a standard, central location defined
        by `internal_certs_location`. For a local, single-host deployment of
        JupyterHub, this should suffice. If, however, singleuser notebooks
        are spawned on other hosts, `.move_certs` should be overridden to move
        these files appropriately. This could mean using `scp` to copy them
        to another host, moving them to a volume mounted in a docker container,
        or exporting them as a secret in kubernetes.
        """
        return paths

    def get_args(self):
        if False:
            while True:
                i = 10
        "Return the arguments to be passed after self.cmd\n\n        Doesn't expect shell expansion to happen.\n\n        .. versionchanged:: 2.0\n            Prior to 2.0, JupyterHub passed some options such as\n            ip, port, and default_url to the command-line.\n            JupyterHub 2.0 no longer builds any CLI args\n            other than `Spawner.cmd` and `Spawner.args`.\n            All values that come from jupyterhub itself\n            will be passed via environment variables.\n        "
        return self.args

    def run_pre_spawn_hook(self):
        if False:
            print('Hello World!')
        'Run the pre_spawn_hook if defined'
        if self.pre_spawn_hook:
            return self.pre_spawn_hook(self)

    def run_post_stop_hook(self):
        if False:
            i = 10
            return i + 15
        'Run the post_stop_hook if defined'
        if self.post_stop_hook is not None:
            try:
                return self.post_stop_hook(self)
            except Exception:
                self.log.exception('post_stop_hook failed with exception: %s', self)

    async def run_auth_state_hook(self, auth_state):
        """Run the auth_state_hook if defined"""
        if self.auth_state_hook is not None:
            await maybe_future(self.auth_state_hook(self, auth_state))

    @property
    def _progress_url(self):
        if False:
            i = 10
            return i + 15
        return self.user.progress_url(self.name)

    async def _generate_progress(self):
        """Private wrapper of progress generator

        This method is always an async generator and will always yield at least one event.
        """
        if not self._spawn_pending:
            self.log.warning("Spawn not pending, can't generate progress for %s", self._log_name)
            return
        yield {'progress': 0, 'message': 'Server requested'}
        async with aclosing(self.progress()) as progress:
            async for event in progress:
                yield event

    async def progress(self):
        """Async generator for progress events

        Must be an async generator

        Should yield messages of the form:

        ::

          {
            "progress": 80, # integer, out of 100
            "message": text, # text message (will be escaped for HTML)
            "html_message": html_text, # optional html-formatted message (may have links)
          }

        In HTML contexts, html_message will be displayed instead of message if present.
        Progress will be updated if defined.
        To update messages without progress omit the progress field.

        .. versionadded:: 0.9
        """
        yield {'progress': 50, 'message': 'Spawning server...'}

    async def start(self):
        """Start the single-user server

        Returns:
          (str, int): the (ip, port) where the Hub can connect to the server.

        .. versionchanged:: 0.7
            Return ip, port instead of setting on self.user.server directly.
        """
        raise NotImplementedError('Override in subclass. Must be a coroutine.')

    async def stop(self, now=False):
        """Stop the single-user server

        If `now` is False (default), shutdown the server as gracefully as possible,
        e.g. starting with SIGINT, then SIGTERM, then SIGKILL.
        If `now` is True, terminate the server immediately.

        The coroutine should return when the single-user server process is no longer running.

        Must be a coroutine.
        """
        raise NotImplementedError('Override in subclass. Must be a coroutine.')

    async def poll(self):
        """Check if the single-user process is running

        Returns:
          None if single-user process is running.
          Integer exit status (0 if unknown), if it is not running.

        State transitions, behavior, and return response:

        - If the Spawner has not been initialized (neither loaded state, nor called start),
          it should behave as if it is not running (status=0).
        - If the Spawner has not finished starting,
          it should behave as if it is running (status=None).

        Design assumptions about when `poll` may be called:

        - On Hub launch: `poll` may be called before `start` when state is loaded on Hub launch.
          `poll` should return exit status 0 (unknown) if the Spawner has not been initialized via
          `load_state` or `start`.
        - If `.start()` is async: `poll` may be called during any yielded portions of the `start`
          process. `poll` should return None when `start` is yielded, indicating that the `start`
          process has not yet completed.

        """
        raise NotImplementedError('Override in subclass. Must be a coroutine.')

    def delete_forever(self):
        if False:
            for i in range(10):
                print('nop')
        'Called when a user or server is deleted.\n\n        This can do things like request removal of resources such as persistent storage.\n        Only called on stopped spawners, and is usually the last action ever taken for the user.\n\n        Will only be called once on each Spawner, immediately prior to removal.\n\n        Stopping a server does *not* call this method.\n        '

    def add_poll_callback(self, callback, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Add a callback to fire when the single-user server stops'
        if args or kwargs:
            cb = callback
            callback = lambda : cb(*args, **kwargs)
        self._callbacks.append(callback)

    def stop_polling(self):
        if False:
            i = 10
            return i + 15
        "Stop polling for single-user server's running state"
        if self._poll_callback:
            self._poll_callback.stop()
            self._poll_callback = None

    def start_polling(self):
        if False:
            print('Hello World!')
        "Start polling periodically for single-user server's running state.\n\n        Callbacks registered via `add_poll_callback` will fire if/when the server stops.\n        Explicit termination via the stop method will not trigger the callbacks.\n        "
        if self.poll_interval <= 0:
            self.log.debug('Not polling subprocess')
            return
        else:
            self.log.debug('Polling subprocess every %is', self.poll_interval)
        self.stop_polling()
        self._poll_callback = PeriodicCallback(self.poll_and_notify, 1000.0 * self.poll_interval)
        self._poll_callback.start()

    async def poll_and_notify(self):
        """Used as a callback to periodically poll the process and notify any watchers"""
        status = await self.poll()
        if status is None:
            return
        self.stop_polling()
        (self._callbacks, callbacks) = ([], self._callbacks)
        for callback in callbacks:
            try:
                await maybe_future(callback())
            except Exception:
                self.log.exception('Unhandled error in poll callback for %s', self)
        return status
    death_interval = Float(0.1)

    async def wait_for_death(self, timeout=10):
        """Wait for the single-user server to die, up to timeout seconds"""

        async def _wait_for_death():
            status = await self.poll()
            return status is not None
        try:
            r = await exponential_backoff(_wait_for_death, f'Process did not die in {timeout} seconds', start_wait=self.death_interval, timeout=timeout)
            return r
        except AnyTimeoutError:
            return False

def _try_setcwd(path):
    if False:
        print('Hello World!')
    'Try to set CWD to path, walking up until a valid directory is found.\n\n    If no valid directory is found, a temp directory is created and cwd is set to that.\n    '
    while path != '/':
        try:
            os.chdir(path)
        except OSError as e:
            exc = e
            print(f"Couldn't set CWD to {path} ({e})", file=sys.stderr)
            (path, _) = os.path.split(path)
        else:
            return
    print("Couldn't set CWD at all (%s), using temp dir" % exc, file=sys.stderr)
    td = mkdtemp()
    os.chdir(td)

def set_user_setuid(username, chdir=True):
    if False:
        for i in range(10):
            print('nop')
    "Return a preexec_fn for spawning a single-user server as a particular user.\n\n    Returned preexec_fn will set uid/gid, and attempt to chdir to the target user's\n    home directory.\n    "
    import pwd
    user = pwd.getpwnam(username)
    uid = user.pw_uid
    gid = user.pw_gid
    home = user.pw_dir
    gids = os.getgrouplist(username, gid)

    def preexec():
        if False:
            print('Hello World!')
        "Set uid/gid of current process\n\n        Executed after fork but before exec by python.\n\n        Also try to chdir to the user's home directory.\n        "
        os.setgid(gid)
        try:
            os.setgroups(gids)
        except Exception as e:
            print('Failed to set groups %s' % e, file=sys.stderr)
        os.setuid(uid)
        if chdir:
            _try_setcwd(home)
    return preexec

class LocalProcessSpawner(Spawner):
    """
    A Spawner that uses `subprocess.Popen` to start single-user servers as local processes.

    Requires local UNIX users matching the authenticated users to exist.
    Does not work on Windows.

    This is the default spawner for JupyterHub.

    Note: This spawner does not implement CPU / memory guarantees and limits.
    """
    interrupt_timeout = Integer(10, help='\n        Seconds to wait for single-user server process to halt after SIGINT.\n\n        If the process has not exited cleanly after this many seconds, a SIGTERM is sent.\n        ').tag(config=True)
    term_timeout = Integer(5, help='\n        Seconds to wait for single-user server process to halt after SIGTERM.\n\n        If the process does not exit cleanly after this many seconds of SIGTERM, a SIGKILL is sent.\n        ').tag(config=True)
    kill_timeout = Integer(5, help='\n        Seconds to wait for process to halt after SIGKILL before giving up.\n\n        If the process does not exit cleanly after this many seconds of SIGKILL, it becomes a zombie\n        process. The hub process will log a warning and then give up.\n        ').tag(config=True)
    popen_kwargs = Dict(help='Extra keyword arguments to pass to Popen\n\n        when spawning single-user servers.\n\n        For example::\n\n            popen_kwargs = dict(shell=True)\n\n        ').tag(config=True)
    shell_cmd = Command(minlen=0, help="Specify a shell command to launch.\n\n        The single-user command will be appended to this list,\n        so it sould end with `-c` (for bash) or equivalent.\n\n        For example::\n\n            c.LocalProcessSpawner.shell_cmd = ['bash', '-l', '-c']\n\n        to launch with a bash login shell, which would set up the user's own complete environment.\n\n        .. warning::\n\n            Using shell_cmd gives users control over PATH, etc.,\n            which could change what the jupyterhub-singleuser launch command does.\n            Only use this for trusted users.\n        ").tag(config=True)
    proc = Instance(Popen, allow_none=True, help='\n        The process representing the single-user server process spawned for current user.\n\n        Is None if no process has been spawned yet.\n        ')
    pid = Integer(0, help='\n        The process id (pid) of the single-user server process spawned for current user.\n        ')

    def make_preexec_fn(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a function that can be used to set the user id of the spawned process to user with name `name`\n\n        This function can be safely passed to `preexec_fn` of `Popen`\n        '
        return set_user_setuid(name)

    def load_state(self, state):
        if False:
            return 10
        'Restore state about spawned single-user server after a hub restart.\n\n        Local processes only need the process id.\n        '
        super().load_state(state)
        if 'pid' in state:
            self.pid = state['pid']

    def get_state(self):
        if False:
            return 10
        'Save state that is needed to restore this spawner instance after a hub restore.\n\n        Local processes only need the process id.\n        '
        state = super().get_state()
        if self.pid:
            state['pid'] = self.pid
        return state

    def clear_state(self):
        if False:
            while True:
                i = 10
        'Clear stored state about this spawner (pid)'
        super().clear_state()
        self.pid = 0

    def user_env(self, env):
        if False:
            for i in range(10):
                print('nop')
        'Augment environment of spawned process with user specific env variables.'
        import pwd
        env['USER'] = self.user.name
        home = pwd.getpwnam(self.user.name).pw_dir
        shell = pwd.getpwnam(self.user.name).pw_shell
        if home:
            env['HOME'] = home
        if shell:
            env['SHELL'] = shell
        return env

    def get_env(self):
        if False:
            i = 10
            return i + 15
        'Get the complete set of environment variables to be set in the spawned process.'
        env = super().get_env()
        env = self.user_env(env)
        return env

    async def move_certs(self, paths):
        """Takes cert paths, moves and sets ownership for them

        Arguments:
            paths (dict): a list of paths for key, cert, and CA

        Returns:
            dict: a list (potentially altered) of paths for key, cert,
            and CA

        Stage certificates into a private home directory
        and make them readable by the user.
        """
        import pwd
        key = paths['keyfile']
        cert = paths['certfile']
        ca = paths['cafile']
        user = pwd.getpwnam(self.user.name)
        uid = user.pw_uid
        gid = user.pw_gid
        home = user.pw_dir
        hub_dir = f'{home}/.jupyterhub'
        out_dir = f'{hub_dir}/jupyterhub-certs'
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, 448, exist_ok=True)
        shutil.move(paths['keyfile'], out_dir)
        shutil.move(paths['certfile'], out_dir)
        shutil.copy(paths['cafile'], out_dir)
        key = os.path.join(out_dir, os.path.basename(paths['keyfile']))
        cert = os.path.join(out_dir, os.path.basename(paths['certfile']))
        ca = os.path.join(out_dir, os.path.basename(paths['cafile']))
        for f in [hub_dir, out_dir, key, cert, ca]:
            shutil.chown(f, user=uid, group=gid)
        return {'keyfile': key, 'certfile': cert, 'cafile': ca}

    async def start(self):
        """Start the single-user server."""
        if self.port == 0:
            self.port = random_port()
        cmd = []
        env = self.get_env()
        cmd.extend(self.cmd)
        cmd.extend(self.get_args())
        if self.shell_cmd:
            cmd = self.shell_cmd + [' '.join((shlex.quote(s) for s in cmd))]
        self.log.info('Spawning %s', ' '.join((shlex.quote(s) for s in cmd)))
        popen_kwargs = dict(preexec_fn=self.make_preexec_fn(self.user.name), start_new_session=True)
        popen_kwargs.update(self.popen_kwargs)
        popen_kwargs['env'] = env
        try:
            self.proc = Popen(cmd, **popen_kwargs)
        except PermissionError:
            script = shutil.which(cmd[0]) or cmd[0]
            self.log.error('Permission denied trying to run %r. Does %s have access to this file?', script, self.user.name)
            raise
        self.pid = self.proc.pid
        return (self.ip or '127.0.0.1', self.port)

    async def poll(self):
        """Poll the spawned process to see if it is still running.

        If the process is still running, we return None. If it is not running,
        we return the exit code of the process if we have access to it, or 0 otherwise.
        """
        if self.proc is not None:
            status = self.proc.poll()
            if status is not None:
                with self.proc:
                    self.clear_state()
            return status
        if not self.pid:
            self.clear_state()
            return 0
        if os.name == 'nt':
            alive = psutil.pid_exists(self.pid)
        else:
            alive = await self._signal(0)
        if not alive:
            self.clear_state()
            return 0
        else:
            return None

    async def _signal(self, sig):
        """Send given signal to a single-user server's process.

        Returns True if the process still exists, False otherwise.

        The hub process is assumed to have enough privileges to do this (e.g. root).
        """
        try:
            os.kill(self.pid, sig)
        except ProcessLookupError:
            return False
        except OSError as e:
            raise
        return True

    async def stop(self, now=False):
        """Stop the single-user server process for the current user.

        If `now` is False (default), shutdown the server as gracefully as possible,
        e.g. starting with SIGINT, then SIGTERM, then SIGKILL.
        If `now` is True, terminate the server immediately.

        The coroutine should return when the process is no longer running.
        """
        if not now:
            status = await self.poll()
            if status is not None:
                return
            self.log.debug('Interrupting %i', self.pid)
            await self._signal(signal.SIGINT)
            await self.wait_for_death(self.interrupt_timeout)
        status = await self.poll()
        if status is not None:
            return
        self.log.debug('Terminating %i', self.pid)
        await self._signal(signal.SIGTERM)
        await self.wait_for_death(self.term_timeout)
        status = await self.poll()
        if status is not None:
            return
        self.log.debug('Killing %i', self.pid)
        await self._signal(signal.SIGKILL)
        await self.wait_for_death(self.kill_timeout)
        status = await self.poll()
        if status is None:
            self.log.warning('Process %i never died', self.pid)

class SimpleLocalProcessSpawner(LocalProcessSpawner):
    """
    A version of LocalProcessSpawner that doesn't require users to exist on
    the system beforehand.

    Only use this for testing.

    Note: DO NOT USE THIS FOR PRODUCTION USE CASES! It is very insecure, and
    provides absolutely no isolation between different users!
    """
    home_dir_template = Unicode('/tmp/{username}', config=True, help='\n        Template to expand to set the user home.\n        {username} is expanded to the jupyterhub username.\n        ')
    home_dir = Unicode(help='The home directory for the user')

    @default('home_dir')
    def _default_home_dir(self):
        if False:
            while True:
                i = 10
        return self.home_dir_template.format(username=self.user.name)

    def make_preexec_fn(self, name):
        if False:
            return 10
        home = self.home_dir

        def preexec():
            if False:
                return 10
            try:
                os.makedirs(home, 493, exist_ok=True)
                os.chdir(home)
            except Exception as e:
                self.log.exception('Error in preexec for %s', name)
        return preexec

    def user_env(self, env):
        if False:
            for i in range(10):
                print('nop')
        env['USER'] = self.user.name
        env['HOME'] = self.home_dir
        env['SHELL'] = '/bin/bash'
        return env

    def move_certs(self, paths):
        if False:
            for i in range(10):
                print('nop')
        'No-op for installing certs.'
        return paths