"""
A non-blocking REST API for Salt
================================

.. py:currentmodule:: salt.netapi.rest_tornado.saltnado

:depends:   - tornado Python module

:configuration: All authentication is done through Salt's :ref:`external auth
    <acl-eauth>` system which requires additional configuration not described
    here.


In order to run rest_tornado with the salt-master
add the following to the Salt master config file.

.. code-block:: yaml

    rest_tornado:
        # can be any port
        port: 8000
        # address to bind to (defaults to 0.0.0.0)
        address: 0.0.0.0
        # socket backlog
        backlog: 128
        ssl_crt: /etc/pki/api/certs/server.crt
        # no need to specify ssl_key if cert and key
        # are in one single file
        ssl_key: /etc/pki/api/certs/server.key
        debug: False
        disable_ssl: False
        webhook_disable_auth: False
        cors_origin: null

.. _rest_tornado-auth:

Authentication
--------------

Authentication is performed by passing a session token with each request.
Tokens are generated via the :py:class:`SaltAuthHandler` URL.

The token may be sent in one of two ways:

* Include a custom header named :mailheader:`X-Auth-Token`.
* Sent via a cookie. This option is a convenience for HTTP clients that
  automatically handle cookie support (such as browsers).

.. seealso:: You can bypass the session handling via the :py:class:`RunSaltAPIHandler` URL.

CORS
----

rest_tornado supports Cross-site HTTP requests out of the box. It is by default
deactivated and controlled by the `cors_origin` config key.

You can allow all origins by settings `cors_origin` to `*`.

You can allow only one origin with this configuration:

.. code-block:: yaml

    rest_tornado:
        cors_origin: http://salt.yourcompany.com

You can also be more specific and select only a few allowed origins by using
a list. For example:

.. code-block:: yaml

    rest_tornado:
        cors_origin:
            - http://salt.yourcompany.com
            - http://salt-preprod.yourcampany.com

The format for origin are full URL, with both scheme and port if not standard.

In this case, rest_tornado will check if the Origin header is in the allowed
list if it's the case allow the origin. Else it will returns nothing,
effectively preventing the origin to make request.

For reference, CORS is a mechanism used by browser to allow (or disallow)
requests made from browser from a different origin than salt-api. It's
complementary to Authentication and mandatory only if you plan to use
a salt client developed as a Javascript browser application.

Usage
-----

Commands are sent to a running Salt master via this module by sending HTTP
requests to the URLs detailed below.

.. admonition:: Content negotiation

    This REST interface is flexible in what data formats it will accept as well
    as what formats it will return (e.g., JSON, YAML, x-www-form-urlencoded).

    * Specify the format of data in the request body by including the
      :mailheader:`Content-Type` header.
    * Specify the desired data format for the response body with the
      :mailheader:`Accept` header.

Data sent in :http:method:`post` and :http:method:`put` requests  must be in
the format of a list of lowstate dictionaries. This allows multiple commands to
be executed in a single HTTP request.

.. glossary::

    lowstate
        A dictionary containing various keys that instruct Salt which command
        to run, where that command lives, any parameters for that command, any
        authentication credentials, what returner to use, etc.

        Salt uses the lowstate data format internally in many places to pass
        command data between functions. Salt also uses lowstate for the
        :ref:`LocalClient() <python-api>` Python API interface.

The following example (in JSON format) causes Salt to execute two commands::

    [{
        "client": "local",
        "tgt": "*",
        "fun": "test.fib",
        "arg": ["10"]
    },
    {
        "client": "runner",
        "fun": "jobs.lookup_jid",
        "jid": "20130603122505459265"
    }]

Multiple commands in a Salt API request will be executed in serial and makes
no guarantees that all commands will run. Meaning that if test.fib (from the
example above) had an exception, the API would still execute "jobs.lookup_jid".

Responses to these lowstates are an in-order list of dicts containing the
return data, a yaml response could look like::

    - ms-1: true
      ms-2: true
    - ms-1: foo
      ms-2: bar

In the event of an exception while executing a command the return for that lowstate
will be a string, for example if no minions matched the first lowstate we would get
a return like::

    - No minions matched the target. No command was sent, no jid was assigned.
    - ms-1: true
      ms-2: true

.. admonition:: x-www-form-urlencoded

    Sending JSON or YAML in the request body is simple and most flexible,
    however sending data in urlencoded format is also supported with the
    caveats below. It is the default format for HTML forms, many JavaScript
    libraries, and the :command:`curl` command.

    For example, the equivalent to running ``salt '*' test.ping`` is sending
    ``fun=test.ping&arg&client=local&tgt=*`` in the HTTP request body.

    Caveats:

    * Only a single command may be sent per HTTP request.
    * Repeating the ``arg`` parameter multiple times will cause those
      parameters to be combined into a single list.

      Note, some popular frameworks and languages (notably jQuery, PHP, and
      Ruby on Rails) will automatically append empty brackets onto repeated
      parameters. E.g., ``arg=one``, ``arg=two`` will be sent as ``arg[]=one``,
      ``arg[]=two``. This is not supported; send JSON or YAML instead.


.. |req_token| replace:: a session token from :py:class:`~SaltAuthHandler`.
.. |req_accept| replace:: the desired response format.
.. |req_ct| replace:: the format of the request body.

.. |res_ct| replace:: the format of the response body; depends on the
    :mailheader:`Accept` request header.

.. |200| replace:: success
.. |400| replace:: bad request
.. |401| replace:: authentication required
.. |406| replace:: requested Content-Type not available
.. |500| replace:: internal server error
"""
import cgi
import fnmatch
import logging
import time
from collections import defaultdict
from copy import copy
import tornado.escape
import tornado.gen
import tornado.httpserver
import tornado.ioloop
import tornado.web
from tornado.concurrent import Future
import salt.auth
import salt.client
import salt.netapi
import salt.runner
import salt.utils.args
import salt.utils.event
import salt.utils.json
import salt.utils.minions
import salt.utils.yaml
from salt.exceptions import AuthenticationError, AuthorizationError, EauthAuthenticationError
from salt.utils.event import tagify
_json = salt.utils.json.import_json()
log = logging.getLogger(__name__)

def _json_dumps(obj, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Invoke salt.utils.json.dumps using the alternate json module loaded using\n    salt.utils.json.import_json(). This ensures that we properly encode any\n    strings in the object before we perform the serialization.\n    '
    return salt.utils.json.dumps(obj, _json_module=_json, **kwargs)
AUTH_TOKEN_HEADER = 'X-Auth-Token'
AUTH_COOKIE_NAME = 'session_id'

class TimeoutException(Exception):
    pass

class Any(Future):
    """
    Future that wraps other futures to "block" until one is done
    """

    def __init__(self, futures):
        if False:
            print('Hello World!')
        super().__init__()
        for future in futures:
            future.add_done_callback(self.done_callback)

    def done_callback(self, future):
        if False:
            while True:
                i = 10
        if not self.done():
            self.set_result(future)

class EventListener:
    """
    Class responsible for listening to the salt master event bus and updating
    futures. This is the core of what makes this asynchronous, this allows us to do
    non-blocking work in the main processes and "wait" for an event to happen
    """

    def __init__(self, mod_opts, opts):
        if False:
            while True:
                i = 10
        self.mod_opts = mod_opts
        self.opts = opts
        self.event = salt.utils.event.get_event('master', opts['sock_dir'], opts=opts, listen=True, io_loop=tornado.ioloop.IOLoop.current())
        self.tag_map = defaultdict(list)
        self.request_map = defaultdict(list)
        self.timeout_map = {}
        self.event.set_event_handler(self._handle_event_socket_recv)

    def clean_by_request(self, request):
        if False:
            return 10
        '\n        Remove all futures that were waiting for request `request` since it is done waiting\n        '
        if request not in self.request_map:
            return
        for (tag, matcher, future) in self.request_map[request]:
            self._timeout_future(tag, matcher, future)
            if future in self.timeout_map:
                tornado.ioloop.IOLoop.current().remove_timeout(self.timeout_map[future])
                del self.timeout_map[future]
        del self.request_map[request]

    @staticmethod
    def prefix_matcher(mtag, tag):
        if False:
            while True:
                i = 10
        if mtag is None or tag is None:
            raise TypeError('mtag or tag can not be None')
        return mtag.startswith(tag)

    @staticmethod
    def exact_matcher(mtag, tag):
        if False:
            return 10
        if mtag is None or tag is None:
            raise TypeError('mtag or tag can not be None')
        return mtag == tag

    def get_event(self, request, tag='', matcher=prefix_matcher.__func__, callback=None, timeout=None):
        if False:
            while True:
                i = 10
        '\n        Get an event (asynchronous of course) return a future that will get it later\n        '
        future = Future()
        if callback is not None:

            def handle_future(future):
                if False:
                    return 10
                tornado.ioloop.IOLoop.current().add_callback(callback, future)
            future.add_done_callback(handle_future)
        self.tag_map[tag, matcher].append(future)
        self.request_map[request].append((tag, matcher, future))
        if timeout:
            timeout_future = tornado.ioloop.IOLoop.current().call_later(timeout, self._timeout_future, tag, matcher, future)
            self.timeout_map[future] = timeout_future
        return future

    def _timeout_future(self, tag, matcher, future):
        if False:
            i = 10
            return i + 15
        '\n        Timeout a specific future\n        '
        if (tag, matcher) not in self.tag_map:
            return
        if not future.done():
            future.set_exception(TimeoutException())
        if future in self.tag_map[tag, matcher] and future.done():
            self.tag_map[tag, matcher].remove(future)
        if len(self.tag_map[tag, matcher]) == 0:
            del self.tag_map[tag, matcher]

    def _handle_event_socket_recv(self, raw):
        if False:
            return 10
        '\n        Callback for events on the event sub socket\n        '
        (mtag, data) = self.event.unpack(raw)
        for ((tag, matcher), futures) in self.tag_map.items():
            try:
                is_matched = matcher(mtag, tag)
            except Exception:
                log.error('Failed to run a matcher.', exc_info=True)
                is_matched = False
            if not is_matched:
                continue
            for future in futures:
                if future.done():
                    continue
                future.set_result({'data': data, 'tag': mtag})
                self.tag_map[tag, matcher].remove(future)
                if future in self.timeout_map:
                    tornado.ioloop.IOLoop.current().remove_timeout(self.timeout_map[future])
                    del self.timeout_map[future]

class BaseSaltAPIHandler(tornado.web.RequestHandler):
    ct_out_map = (('application/json', _json_dumps), ('application/x-yaml', salt.utils.yaml.safe_dump))

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self._auto_finish = False

    def _verify_client(self, low):
        if False:
            while True:
                i = 10
        '\n        Verify that the client is in fact one we have\n        '
        if 'client' not in low or low.get('client') not in self.saltclients:
            self.set_status(400)
            self.write('400 Invalid Client: Client not found in salt clients')
            self.finish()
            return False
        return True

    def initialize(self):
        if False:
            i = 10
            return i + 15
        '\n        Initialize the handler before requests are called\n        '
        if not hasattr(self.application, 'event_listener'):
            log.debug('init a listener')
            self.application.event_listener = EventListener(self.application.mod_opts, self.application.opts)
        if not hasattr(self, 'saltclients'):
            local_client = salt.client.get_local_client(mopts=self.application.opts)
            self.saltclients = {'local': local_client.run_job_async, 'local_async': local_client.run_job_async, 'runner': salt.runner.RunnerClient(opts=self.application.opts).cmd_async, 'runner_async': None}
        if not hasattr(self, 'ckminions'):
            self.ckminions = salt.utils.minions.CkMinions(self.application.opts)

    @property
    def token(self):
        if False:
            i = 10
            return i + 15
        '\n        The token used for the request\n        '
        if AUTH_TOKEN_HEADER in self.request.headers:
            return self.request.headers[AUTH_TOKEN_HEADER]
        else:
            return self.get_cookie(AUTH_COOKIE_NAME)

    def _verify_auth(self):
        if False:
            print('Hello World!')
        "\n        Boolean whether the request is auth'd\n        "
        if self.token:
            token_dict = self.application.auth.get_tok(self.token)
            if token_dict and token_dict.get('expire', 0) > time.time():
                return True
        return False

    def prepare(self):
        if False:
            print('Hello World!')
        '\n        Run before get/posts etc. Pre-flight checks:\n            - verify that we can speak back to them (compatible accept header)\n        '
        accept_header = self.request.headers.get('Accept', '*/*')
        parsed_accept_header = [cgi.parse_header(h)[0] for h in accept_header.split(',')]

        def find_acceptable_content_type(parsed_accept_header):
            if False:
                print('Hello World!')
            for media_range in parsed_accept_header:
                for (content_type, dumper) in self.ct_out_map:
                    if fnmatch.fnmatch(content_type, media_range):
                        return (content_type, dumper)
            return (None, None)
        (content_type, dumper) = find_acceptable_content_type(parsed_accept_header)
        if not content_type:
            self.send_error(406)
        self.content_type = content_type
        self.dumper = dumper
        self.start = time.time()
        self.connected = True
        self.lowstate = self._get_lowstate()

    def timeout_futures(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        timeout a session\n        '
        self.application.event_listener.clean_by_request(self)

    def on_finish(self):
        if False:
            return 10
        '\n        When the job has been done, lets cleanup\n        '
        self.timeout_futures()
        del self.saltclients

    def on_connection_close(self):
        if False:
            return 10
        '\n        If the client disconnects, lets close out\n        '
        self.finish()

    def serialize(self, data):
        if False:
            return 10
        '\n        Serlialize the output based on the Accept header\n        '
        self.set_header('Content-Type', self.content_type)
        return self.dumper(data)

    def _form_loader(self, _):
        if False:
            for i in range(10):
                print('nop')
        '\n        function to get the data from the urlencoded forms\n        ignore the data passed in and just get the args from wherever they are\n        '
        data = {}
        for key in self.request.arguments:
            val = self.get_arguments(key)
            if len(val) == 1:
                data[key] = val[0]
            else:
                data[key] = val
        return data

    def deserialize(self, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deserialize the data based on request content type headers\n        '
        ct_in_map = {'application/x-www-form-urlencoded': self._form_loader, 'application/json': salt.utils.json.loads, 'application/x-yaml': salt.utils.yaml.safe_load, 'text/yaml': salt.utils.yaml.safe_load, 'text/plain': salt.utils.json.loads}
        try:
            (value, parameters) = cgi.parse_header(self.request.headers['Content-Type'])
            return ct_in_map[value](tornado.escape.native_str(data))
        except KeyError:
            self.send_error(406)
        except ValueError:
            self.send_error(400)

    def _get_lowstate(self):
        if False:
            i = 10
            return i + 15
        '\n        Format the incoming data into a lowstate object\n        '
        if not self.request.body:
            return
        data = self.deserialize(self.request.body)
        self.request_payload = copy(data)
        if data and 'arg' in data and (not isinstance(data['arg'], list)):
            data['arg'] = [data['arg']]
        if not isinstance(data, list):
            lowstate = [data]
        else:
            lowstate = data
        return lowstate

    def set_default_headers(self):
        if False:
            return 10
        '\n        Set default CORS headers\n        '
        mod_opts = self.application.mod_opts
        if mod_opts.get('cors_origin'):
            origin = self.request.headers.get('Origin')
            allowed_origin = _check_cors_origin(origin, mod_opts['cors_origin'])
            if allowed_origin:
                self.set_header('Access-Control-Allow-Origin', allowed_origin)

    def options(self, *args, **kwargs):
        if False:
            return 10
        '\n        Return CORS headers for preflight requests\n        '
        request_headers = self.request.headers.get('Access-Control-Request-Headers')
        allowed_headers = request_headers.split(',')
        self.set_header('Access-Control-Allow-Headers', ','.join(allowed_headers))
        self.set_header('Access-Control-Expose-Headers', 'X-Auth-Token')
        self.set_header('Access-Control-Allow-Methods', 'OPTIONS, GET, POST')
        self.set_status(204)
        self.finish()

class SaltAuthHandler(BaseSaltAPIHandler):
    """
    Handler for login requests
    """

    def get(self):
        if False:
            while True:
                i = 10
        '\n        All logins are done over post, this is a parked endpoint\n\n        .. http:get:: /login\n\n            :status 401: |401|\n            :status 406: |406|\n\n        **Example request:**\n\n        .. code-block:: bash\n\n            curl -i localhost:8000/login\n\n        .. code-block:: text\n\n            GET /login HTTP/1.1\n            Host: localhost:8000\n            Accept: application/json\n\n        **Example response:**\n\n        .. code-block:: text\n\n            HTTP/1.1 401 Unauthorized\n            Content-Type: application/json\n            Content-Length: 58\n\n            {"status": "401 Unauthorized", "return": "Please log in"}\n        '
        self.set_status(401)
        self.set_header('WWW-Authenticate', 'Session')
        ret = {'status': '401 Unauthorized', 'return': 'Please log in'}
        self.write(self.serialize(ret))
        self.finish()

    def post(self):
        if False:
            while True:
                i = 10
        '\n        :ref:`Authenticate <rest_tornado-auth>` against Salt\'s eauth system\n\n        .. http:post:: /login\n\n            :reqheader X-Auth-Token: |req_token|\n            :reqheader Accept: |req_accept|\n            :reqheader Content-Type: |req_ct|\n\n            :form eauth: the eauth backend configured for the user\n            :form username: username\n            :form password: password\n\n            :status 200: |200|\n            :status 400: |400|\n            :status 401: |401|\n            :status 406: |406|\n            :status 500: |500|\n\n        **Example request:**\n\n        .. code-block:: bash\n\n            curl -si localhost:8000/login \\\n                    -H "Accept: application/json" \\\n                    -d username=\'saltuser\' \\\n                    -d password=\'saltpass\' \\\n                    -d eauth=\'pam\'\n\n        .. code-block:: text\n\n            POST / HTTP/1.1\n            Host: localhost:8000\n            Content-Length: 42\n            Content-Type: application/x-www-form-urlencoded\n            Accept: application/json\n\n            username=saltuser&password=saltpass&eauth=pam\n\n        **Example response:**\n\n        .. code-block:: text\n\n            HTTP/1.1 200 OK\n            Content-Type: application/json\n            Content-Length: 206\n            X-Auth-Token: 6d1b722e\n            Set-Cookie: session_id=6d1b722e; expires=Sat, 17 Nov 2012 03:23:52 GMT; Path=/\n\n            {"return": {\n                "token": "6d1b722e",\n                "start": 1363805943.776223,\n                "expire": 1363849143.776224,\n                "user": "saltuser",\n                "eauth": "pam",\n                "perms": [\n                    "grains.*",\n                    "status.*",\n                    "sys.*",\n                    "test.*"\n                ]\n            }}\n        '
        try:
            if not isinstance(self.request_payload, dict):
                self.send_error(400)
                return
            creds = {'username': self.request_payload['username'], 'password': self.request_payload['password'], 'eauth': self.request_payload['eauth']}
        except KeyError:
            self.send_error(400)
            return
        token = self.application.auth.mk_token(creds)
        if 'token' not in token:
            self.send_error(401)
            return
        self.set_cookie(AUTH_COOKIE_NAME, token['token'])
        try:
            eauth = self.application.opts['external_auth'][token['eauth']]
            perms = salt.netapi.sum_permissions(token, eauth)
            perms = salt.netapi.sorted_permissions(perms)
        except KeyError:
            self.send_error(401)
            return
        except (AttributeError, IndexError):
            log.debug("Configuration for external_auth malformed for eauth '%s', and user '%s'.", token.get('eauth'), token.get('name'), exc_info=True)
            self.send_error(500)
            return
        ret = {'return': [{'token': token['token'], 'expire': token['expire'], 'start': token['start'], 'user': token['name'], 'eauth': token['eauth'], 'perms': perms}]}
        self.write(self.serialize(ret))
        self.finish()

class SaltAPIHandler(BaseSaltAPIHandler):
    """
    Main API handler for base "/"
    """

    def get(self):
        if False:
            while True:
                i = 10
        '\n        An endpoint to determine salt-api capabilities\n\n        .. http:get:: /\n\n            :reqheader Accept: |req_accept|\n\n            :status 200: |200|\n            :status 401: |401|\n            :status 406: |406|\n\n        **Example request:**\n\n        .. code-block:: bash\n\n            curl -i localhost:8000\n\n        .. code-block:: text\n\n            GET / HTTP/1.1\n            Host: localhost:8000\n            Accept: application/json\n\n        **Example response:**\n\n        .. code-block:: text\n\n            HTTP/1.1 200 OK\n            Content-Type: application/json\n            Content-Legnth: 83\n\n            {"clients": ["local", "local_async", "runner", "runner_async"], "return": "Welcome"}\n        '
        ret = {'clients': list(self.saltclients.keys()), 'return': 'Welcome'}
        self.write(self.serialize(ret))
        self.finish()

    @tornado.gen.coroutine
    def post(self):
        if False:
            while True:
                i = 10
        '\n        Send one or more Salt commands (lowstates) in the request body\n\n        .. http:post:: /\n\n            :reqheader X-Auth-Token: |req_token|\n            :reqheader Accept: |req_accept|\n            :reqheader Content-Type: |req_ct|\n\n            :resheader Content-Type: |res_ct|\n\n            :status 200: |200|\n            :status 401: |401|\n            :status 406: |406|\n\n            :term:`lowstate` data describing Salt commands must be sent in the\n            request body.\n\n        **Example request:**\n\n        .. code-block:: bash\n\n            curl -si https://localhost:8000 \\\n                    -H "Accept: application/x-yaml" \\\n                    -H "X-Auth-Token: d40d1e1e" \\\n                    -d client=local \\\n                    -d tgt=\'*\' \\\n                    -d fun=\'test.ping\' \\\n                    -d arg\n\n        .. code-block:: text\n\n            POST / HTTP/1.1\n            Host: localhost:8000\n            Accept: application/x-yaml\n            X-Auth-Token: d40d1e1e\n            Content-Length: 36\n            Content-Type: application/x-www-form-urlencoded\n\n            fun=test.ping&arg&client=local&tgt=*\n\n        **Example response:**\n\n        Responses are an in-order list of the lowstate\'s return data. In the\n        event of an exception running a command the return will be a string\n        instead of a mapping.\n\n        .. code-block:: text\n\n            HTTP/1.1 200 OK\n            Content-Length: 200\n            Allow: GET, HEAD, POST\n            Content-Type: application/x-yaml\n\n            return:\n            - ms-0: true\n                ms-1: true\n                ms-2: true\n                ms-3: true\n                ms-4: true\n\n        .. admonition:: multiple commands\n\n            Note that if multiple :term:`lowstate` structures are sent, the Salt\n            API will execute them in serial, and will not stop execution upon failure\n            of a previous job. If you need to have commands executed in order and\n            stop on failure please use compound-command-execution.\n\n        '
        if not self._verify_auth():
            self.redirect('/login')
            return
        self.disbatch()

    @tornado.gen.coroutine
    def disbatch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Disbatch all lowstates to the appropriate clients\n        '
        ret = []
        for low in self.lowstate:
            if not self._verify_client(low):
                return
            if self.token is not None and 'token' not in low:
                low['token'] = self.token
            if not ('token' in low or ('username' in low and 'password' in low and ('eauth' in low))):
                ret.append('Failed to authenticate')
                break
            try:
                chunk_ret = (yield getattr(self, '_disbatch_{}'.format(low['client']))(low))
                ret.append(chunk_ret)
            except (AuthenticationError, AuthorizationError, EauthAuthenticationError):
                ret.append('Failed to authenticate')
                break
            except Exception as ex:
                ret.append('Unexpected exception while handling request: {}'.format(ex))
                log.error('Unexpected exception while handling request:', exc_info=True)
        try:
            self.write(self.serialize({'return': ret}))
            self.finish()
        except RuntimeError as exc:
            log.exception('Encountered Runtime Error')

    @tornado.gen.coroutine
    def get_minion_returns(self, events, is_finished, is_timed_out, min_wait_time, minions):
        if False:
            for i in range(10):
                print('nop')

        def more_todo():
            if False:
                i = 10
                return i + 15
            '\n            Check if there are any more minions we are waiting on returns from\n            '
            return any((x is False for x in minions.values()))
        chunk_ret = {}
        while True:
            to_wait = events + [is_finished, is_timed_out]
            if not min_wait_time.done():
                to_wait += [min_wait_time]

            def cancel_inflight_futures():
                if False:
                    print('Hello World!')
                for event in to_wait:
                    if not event.done() and event is not is_timed_out:
                        event.set_result(None)
            f = (yield Any(to_wait))
            try:
                if f is is_finished or f is is_timed_out:
                    cancel_inflight_futures()
                    raise tornado.gen.Return(chunk_ret)
                elif f is min_wait_time:
                    if not more_todo():
                        cancel_inflight_futures()
                        raise tornado.gen.Return(chunk_ret)
                    continue
                f_result = f.result()
                if f_result['tag'].endswith('/new'):
                    for minion_id in f_result['data']['minions']:
                        if minion_id not in minions:
                            minions[minion_id] = False
                else:
                    chunk_ret[f_result['data']['id']] = f_result['data']['return']
                    minions[f_result['data']['id']] = True
                    if not more_todo() and min_wait_time.done():
                        cancel_inflight_futures()
                        raise tornado.gen.Return(chunk_ret)
            except TimeoutException:
                pass
            finally:
                if f in events:
                    events.remove(f)

    @tornado.gen.coroutine
    def _disbatch_local(self, chunk):
        if False:
            return 10
        '\n        Dispatch local client commands\n        '
        chunk['jid'] = salt.utils.jid.gen_jid(self.application.opts) if not chunk.get('jid', None) else chunk['jid']
        minions = set(self.ckminions.check_minions(chunk['tgt'], chunk.get('tgt_type', 'glob')))

        def subscribe_minion(minion):
            if False:
                return 10
            salt_evt = self.application.event_listener.get_event(self, tag='salt/job/{}/ret/{}'.format(chunk['jid'], minion), matcher=EventListener.exact_matcher)
            syndic_evt = self.application.event_listener.get_event(self, tag='syndic/job/{}/ret/{}'.format(chunk['jid'], minion), matcher=EventListener.exact_matcher)
            return (salt_evt, syndic_evt)
        events = []
        for minion in minions:
            (salt_evt, syndic_evt) = subscribe_minion(minion)
            events.append(salt_evt)
            events.append(syndic_evt)
        f_call = self._format_call_run_job_async(chunk)
        pub_data = (yield self.saltclients['local'](*f_call.get('args', ()), **f_call.get('kwargs', {})))
        if 'jid' not in pub_data:
            for future in events:
                try:
                    future.set_result(None)
                except Exception:
                    pass
            raise tornado.gen.Return('No minions matched the target. No command was sent, no jid was assigned.')
        for minion in list(set(pub_data['minions']) - set(minions)):
            (salt_evt, syndic_evt) = subscribe_minion(minion)
            events.append(salt_evt)
            events.append(syndic_evt)
        minions = {m: False for m in pub_data['minions']}
        min_wait_time = Future()
        min_wait_time.set_result(True)
        if self.application.opts['order_masters']:
            min_wait_time = tornado.gen.sleep(self.application.opts['syndic_wait'])
        is_finished = tornado.gen.Future()
        is_timed_out = tornado.gen.sleep(self.application.opts['gather_job_timeout'])
        tornado.ioloop.IOLoop.current().spawn_callback(self.job_not_running, pub_data['jid'], chunk['tgt'], f_call['kwargs']['tgt_type'], minions, is_finished)
        result = (yield self.get_minion_returns(events=events, is_finished=is_finished, is_timed_out=is_timed_out, min_wait_time=min_wait_time, minions=minions))
        raise tornado.gen.Return(result)

    @tornado.gen.coroutine
    def job_not_running(self, jid, tgt, tgt_type, minions, is_finished):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a future which will complete once jid (passed in) is no longer\n        running on tgt\n        '
        ping_pub_data = (yield self.saltclients['local'](tgt, 'saltutil.find_job', [jid], tgt_type=tgt_type))
        ping_tag = tagify([ping_pub_data['jid'], 'ret'], 'job')
        minion_running = False
        while True:
            try:
                event = self.application.event_listener.get_event(self, tag=ping_tag, timeout=self.application.opts['gather_job_timeout'])
                f = (yield Any([event, is_finished]))
                if f is is_finished:
                    if not event.done():
                        event.set_result(None)
                    raise tornado.gen.Return(True)
                event = f.result()
            except TimeoutException:
                if not minion_running or is_finished.done():
                    raise tornado.gen.Return(True)
                else:
                    ping_pub_data = (yield self.saltclients['local'](tgt, 'saltutil.find_job', [jid], tgt_type=tgt_type))
                    ping_tag = tagify([ping_pub_data['jid'], 'ret'], 'job')
                    minion_running = False
                    continue
            if event['data'].get('return', {}) == {}:
                continue
            if event['data']['id'] not in minions:
                minions[event['data']['id']] = False
            minion_running = True

    @tornado.gen.coroutine
    def _disbatch_local_async(self, chunk):
        if False:
            print('Hello World!')
        '\n        Disbatch local client_async commands\n        '
        f_call = self._format_call_run_job_async(chunk)
        pub_data = (yield self.saltclients['local_async'](*f_call.get('args', ()), **f_call.get('kwargs', {})))
        raise tornado.gen.Return(pub_data)

    @tornado.gen.coroutine
    def _disbatch_runner(self, chunk):
        if False:
            while True:
                i = 10
        '\n        Disbatch runner client commands\n        '
        full_return = chunk.pop('full_return', False)
        pub_data = self.saltclients['runner'](chunk)
        tag = pub_data['tag'] + '/ret'
        try:
            event = (yield self.application.event_listener.get_event(self, tag=tag))
            ret = event if full_return else event['data']['return']
            raise tornado.gen.Return(ret)
        except TimeoutException:
            raise tornado.gen.Return('Timeout waiting for runner to execute')

    @tornado.gen.coroutine
    def _disbatch_runner_async(self, chunk):
        if False:
            while True:
                i = 10
        '\n        Disbatch runner client_async commands\n        '
        pub_data = self.saltclients['runner'](chunk)
        raise tornado.gen.Return(pub_data)

    def _format_call_run_job_async(self, chunk):
        if False:
            while True:
                i = 10
        f_call = salt.utils.args.format_call(salt.client.LocalClient.run_job, chunk, is_class_method=True)
        f_call.get('kwargs', {})['io_loop'] = tornado.ioloop.IOLoop.current()
        return f_call

class MinionSaltAPIHandler(SaltAPIHandler):
    """
    A convenience endpoint for minion related functions
    """

    @tornado.gen.coroutine
    def get(self, mid=None):
        if False:
            i = 10
            return i + 15
        '\n        A convenience URL for getting lists of minions or getting minion\n        details\n\n        .. http:get:: /minions/(mid)\n\n            :reqheader X-Auth-Token: |req_token|\n            :reqheader Accept: |req_accept|\n\n            :status 200: |200|\n            :status 401: |401|\n            :status 406: |406|\n\n        **Example request:**\n\n        .. code-block:: bash\n\n            curl -i localhost:8000/minions/ms-3\n\n        .. code-block:: text\n\n            GET /minions/ms-3 HTTP/1.1\n            Host: localhost:8000\n            Accept: application/x-yaml\n\n        **Example response:**\n\n        .. code-block:: text\n\n            HTTP/1.1 200 OK\n            Content-Length: 129005\n            Content-Type: application/x-yaml\n\n            return:\n            - ms-3:\n                grains.items:\n                    ...\n        '
        if not self._verify_auth():
            self.redirect('/login')
            return
        self.lowstate = [{'client': 'local', 'tgt': mid or '*', 'fun': 'grains.items'}]
        self.disbatch()

    @tornado.gen.coroutine
    def post(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Start an execution command and immediately return the job id\n\n        .. http:post:: /minions\n\n            :reqheader X-Auth-Token: |req_token|\n            :reqheader Accept: |req_accept|\n            :reqheader Content-Type: |req_ct|\n\n            :resheader Content-Type: |res_ct|\n\n            :status 200: |200|\n            :status 401: |401|\n            :status 406: |406|\n\n            :term:`lowstate` data describing Salt commands must be sent in the\n            request body. The ``client`` option will be set to\n            :py:meth:`~salt.client.LocalClient.local_async`.\n\n        **Example request:**\n\n        .. code-block:: bash\n\n            curl -sSi localhost:8000/minions \\\n                -H "Accept: application/x-yaml" \\\n                -d tgt=\'*\' \\\n                -d fun=\'status.diskusage\'\n\n        .. code-block:: text\n\n            POST /minions HTTP/1.1\n            Host: localhost:8000\n            Accept: application/x-yaml\n            Content-Length: 26\n            Content-Type: application/x-www-form-urlencoded\n\n            tgt=*&fun=status.diskusage\n\n        **Example response:**\n\n        .. code-block:: text\n\n            HTTP/1.1 202 Accepted\n            Content-Length: 86\n            Content-Type: application/x-yaml\n\n            return:\n            - jid: \'20130603122505459265\'\n              minions: [ms-4, ms-3, ms-2, ms-1, ms-0]\n        '
        if not self._verify_auth():
            self.redirect('/login')
            return
        for low in self.lowstate:
            if 'client' not in low:
                low['client'] = 'local_async'
                continue
            if low.get('client') != 'local_async':
                self.set_status(400)
                self.write("We don't serve your kind here")
                self.finish()
                return
        self.disbatch()

class JobsSaltAPIHandler(SaltAPIHandler):
    """
    A convenience endpoint for job cache data
    """

    @tornado.gen.coroutine
    def get(self, jid=None):
        if False:
            while True:
                i = 10
        "\n        A convenience URL for getting lists of previously run jobs or getting\n        the return from a single job\n\n        .. http:get:: /jobs/(jid)\n\n            List jobs or show a single job from the job cache.\n\n            :status 200: |200|\n            :status 401: |401|\n            :status 406: |406|\n\n        **Example request:**\n\n        .. code-block:: bash\n\n            curl -i localhost:8000/jobs\n\n        .. code-block:: text\n\n            GET /jobs HTTP/1.1\n            Host: localhost:8000\n            Accept: application/x-yaml\n\n        **Example response:**\n\n        .. code-block:: text\n\n            HTTP/1.1 200 OK\n            Content-Length: 165\n            Content-Type: application/x-yaml\n\n            return:\n            - '20121130104633606931':\n                Arguments:\n                - '3'\n                Function: test.fib\n                Start Time: 2012, Nov 30 10:46:33.606931\n                Target: jerry\n                Target-type: glob\n\n        **Example request:**\n\n        .. code-block:: bash\n\n            curl -i localhost:8000/jobs/20121130104633606931\n\n        .. code-block:: text\n\n            GET /jobs/20121130104633606931 HTTP/1.1\n            Host: localhost:8000\n            Accept: application/x-yaml\n\n        **Example response:**\n\n        .. code-block:: text\n\n            HTTP/1.1 200 OK\n            Content-Length: 73\n            Content-Type: application/x-yaml\n\n            info:\n            - Arguments:\n                - '3'\n                Function: test.fib\n                Minions:\n                - jerry\n                Start Time: 2012, Nov 30 10:46:33.606931\n                Target: '*'\n                Target-type: glob\n                User: saltdev\n                jid: '20121130104633606931'\n            return:\n            - jerry:\n                - - 0\n                - 1\n                - 1\n                - 2\n                - 6.9141387939453125e-06\n        "
        if not self._verify_auth():
            self.redirect('/login')
            return
        if jid:
            self.lowstate = [{'fun': 'jobs.list_job', 'jid': jid, 'client': 'runner'}]
        else:
            self.lowstate = [{'fun': 'jobs.list_jobs', 'client': 'runner'}]
        self.disbatch()

class RunSaltAPIHandler(SaltAPIHandler):
    """
    Endpoint to run commands without normal session handling
    """

    @tornado.gen.coroutine
    def post(self):
        if False:
            return 10
        '\n        Run commands bypassing the :ref:`normal session handling\n        <rest_cherrypy-auth>`\n\n        .. http:post:: /run\n\n            This entry point is primarily for "one-off" commands. Each request\n            must pass full Salt authentication credentials. Otherwise this URL\n            is identical to the :py:meth:`root URL (/) <LowDataAdapter.POST>`.\n\n            :term:`lowstate` data describing Salt commands must be sent in the\n            request body.\n\n            :status 200: |200|\n            :status 401: |401|\n            :status 406: |406|\n\n        **Example request:**\n\n        .. code-block:: bash\n\n            curl -sS localhost:8000/run \\\n                -H \'Accept: application/x-yaml\' \\\n                -d client=\'local\' \\\n                -d tgt=\'*\' \\\n                -d fun=\'test.ping\' \\\n                -d username=\'saltdev\' \\\n                -d password=\'saltdev\' \\\n                -d eauth=\'pam\'\n\n        .. code-block:: text\n\n            POST /run HTTP/1.1\n            Host: localhost:8000\n            Accept: application/x-yaml\n            Content-Length: 75\n            Content-Type: application/x-www-form-urlencoded\n\n            client=local&tgt=*&fun=test.ping&username=saltdev&password=saltdev&eauth=pam\n\n        **Example response:**\n\n        .. code-block:: text\n\n            HTTP/1.1 200 OK\n            Content-Length: 73\n            Content-Type: application/x-yaml\n\n            return:\n            - ms-0: true\n                ms-1: true\n                ms-2: true\n                ms-3: true\n                ms-4: true\n        '
        self.disbatch()

class EventsSaltAPIHandler(SaltAPIHandler):
    """
    Expose the Salt event bus

    The event bus on the Salt master exposes a large variety of things, notably
    when executions are started on the master and also when minions ultimately
    return their results. This URL provides a real-time window into a running
    Salt infrastructure.

    .. seealso:: :ref:`events`
    """

    @tornado.gen.coroutine
    def get(self):
        if False:
            while True:
                i = 10
        '\n        An HTTP stream of the Salt master event bus\n\n        This stream is formatted per the Server Sent Events (SSE) spec. Each\n        event is formatted as JSON.\n\n        .. http:get:: /events\n\n            :status 200: |200|\n            :status 401: |401|\n            :status 406: |406|\n\n        **Example request:**\n\n        .. code-block:: bash\n\n            curl -NsS localhost:8000/events\n\n        .. code-block:: text\n\n            GET /events HTTP/1.1\n            Host: localhost:8000\n\n        **Example response:**\n\n        .. code-block:: text\n\n            HTTP/1.1 200 OK\n            Connection: keep-alive\n            Cache-Control: no-cache\n            Content-Type: text/event-stream;charset=utf-8\n\n            retry: 400\n            data: {\'tag\': \'\', \'data\': {\'minions\': [\'ms-4\', \'ms-3\', \'ms-2\', \'ms-1\', \'ms-0\']}}\n\n            data: {\'tag\': \'20130802115730568475\', \'data\': {\'jid\': \'20130802115730568475\', \'return\': True, \'retcode\': 0, \'success\': True, \'cmd\': \'_return\', \'fun\': \'test.ping\', \'id\': \'ms-1\'}}\n\n        The event stream can be easily consumed via JavaScript:\n\n        .. code-block:: javascript\n\n            <!-- Note, you must be authenticated! -->\n            var source = new EventSource(\'/events\');\n            source.onopen = function() { console.debug(\'opening\') };\n            source.onerror = function(e) { console.debug(\'error!\', e) };\n            source.onmessage = function(e) { console.debug(e.data) };\n\n        Or using CORS:\n\n        .. code-block:: javascript\n\n            var source = new EventSource(\'/events\', {withCredentials: true});\n\n        Some browser clients lack CORS support for the ``EventSource()`` API. Such\n        clients may instead pass the :mailheader:`X-Auth-Token` value as an URL\n        parameter:\n\n        .. code-block:: bash\n\n            curl -NsS localhost:8000/events/6d1b722e\n\n        It is also possible to consume the stream via the shell.\n\n        Records are separated by blank lines; the ``data:`` and ``tag:``\n        prefixes will need to be removed manually before attempting to\n        unserialize the JSON.\n\n        curl\'s ``-N`` flag turns off input buffering which is required to\n        process the stream incrementally.\n\n        Here is a basic example of printing each event as it comes in:\n\n        .. code-block:: bash\n\n            curl -NsS localhost:8000/events |\\\n                    while IFS= read -r line ; do\n                        echo $line\n                    done\n\n        Here is an example of using awk to filter events based on tag:\n\n        .. code-block:: bash\n\n            curl -NsS localhost:8000/events |\\\n                    awk \'\n                        BEGIN { RS=""; FS="\\\\n" }\n                        $1 ~ /^tag: salt\\/job\\/[0-9]+\\/new$/ { print $0 }\n                    \'\n            tag: salt/job/20140112010149808995/new\n            data: {"tag": "salt/job/20140112010149808995/new", "data": {"tgt_type": "glob", "jid": "20140112010149808995", "tgt": "jerry", "_stamp": "2014-01-12_01:01:49.809617", "user": "shouse", "arg": [], "fun": "test.ping", "minions": ["jerry"]}}\n            tag: 20140112010149808995\n            data: {"tag": "20140112010149808995", "data": {"fun_args": [], "jid": "20140112010149808995", "return": true, "retcode": 0, "success": true, "cmd": "_return", "_stamp": "2014-01-12_01:01:49.819316", "fun": "test.ping", "id": "jerry"}}\n        '
        if not self._verify_auth():
            self.redirect('/login')
            return
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Cache-Control', 'no-cache')
        self.set_header('Connection', 'keep-alive')
        self.write('retry: {}\n'.format(400))
        self.flush()
        while True:
            try:
                if not self._verify_auth():
                    log.debug('Token is no longer valid')
                    break
                event = (yield self.application.event_listener.get_event(self))
                self.write('tag: {}\n'.format(event.get('tag', '')))
                self.write('data: {}\n\n'.format(_json_dumps(event)))
                self.flush()
            except TimeoutException:
                break

class WebhookSaltAPIHandler(SaltAPIHandler):
    """
    A generic web hook entry point that fires an event on Salt's event bus

    External services can POST data to this URL to trigger an event in Salt.
    For example, Amazon SNS, Jenkins-CI or Travis-CI, or GitHub web hooks.

    .. note:: Be mindful of security

        Salt's Reactor can run any code. A Reactor SLS that responds to a hook
        event is responsible for validating that the event came from a trusted
        source and contains valid data.

        **This is a generic interface and securing it is up to you!**

        This URL requires authentication however not all external services can
        be configured to authenticate. For this reason authentication can be
        selectively disabled for this URL. Follow best practices -- always use
        SSL, pass a secret key, configure the firewall to only allow traffic
        from a known source, etc.

    The event data is taken from the request body. The
    :mailheader:`Content-Type` header is respected for the payload.

    The event tag is prefixed with ``salt/netapi/hook`` and the URL path is
    appended to the end. For example, a ``POST`` request sent to
    ``/hook/mycompany/myapp/mydata`` will produce a Salt event with the tag
    ``salt/netapi/hook/mycompany/myapp/mydata``.

    The following is an example ``.travis.yml`` file to send notifications to
    Salt of successful test runs:

    .. code-block:: yaml

        language: python
        script: python -m unittest tests
        after_success:
            - 'curl -sS http://saltapi-url.example.com:8000/hook/travis/build/success -d branch="${TRAVIS_BRANCH}" -d commit="${TRAVIS_COMMIT}"'

    .. seealso:: :ref:`Events <events>`, :ref:`Reactor <reactor>`
    """

    def post(self, tag_suffix=None):
        if False:
            while True:
                i = 10
        '\n        Fire an event in Salt with a custom event tag and data\n\n        .. http:post:: /hook\n\n            :status 200: |200|\n            :status 401: |401|\n            :status 406: |406|\n            :status 413: request body is too large\n\n        **Example request:**\n\n        .. code-block:: bash\n\n            curl -sS localhost:8000/hook -d foo=\'Foo!\' -d bar=\'Bar!\'\n\n        .. code-block:: text\n\n            POST /hook HTTP/1.1\n            Host: localhost:8000\n            Content-Length: 16\n            Content-Type: application/x-www-form-urlencoded\n\n            foo=Foo&bar=Bar!\n\n        **Example response**:\n\n        .. code-block:: text\n\n            HTTP/1.1 200 OK\n            Content-Length: 14\n            Content-Type: application/json\n\n            {"success": true}\n\n        As a practical example, an internal continuous-integration build\n        server could send an HTTP POST request to the URL\n        ``http://localhost:8000/hook/mycompany/build/success`` which contains\n        the result of a build and the SHA of the version that was built as\n        JSON. That would then produce the following event in Salt that could be\n        used to kick off a deployment via Salt\'s Reactor:\n\n        .. code-block:: text\n\n            Event fired at Fri Feb 14 17:40:11 2014\n            *************************\n            Tag: salt/netapi/hook/mycompany/build/success\n            Data:\n            {\'_stamp\': \'2014-02-14_17:40:11.440996\',\n                \'headers\': {\n                    \'X-My-Secret-Key\': \'F0fAgoQjIT@W\',\n                    \'Content-Length\': \'37\',\n                    \'Content-Type\': \'application/json\',\n                    \'Host\': \'localhost:8000\',\n                    \'Remote-Addr\': \'127.0.0.1\'},\n                \'post\': {\'revision\': \'aa22a3c4b2e7\', \'result\': True}}\n\n        Salt\'s Reactor could listen for the event:\n\n        .. code-block:: yaml\n\n            reactor:\n              - \'salt/netapi/hook/mycompany/build/*\':\n                - /srv/reactor/react_ci_builds.sls\n\n        And finally deploy the new build:\n\n        .. code-block:: jinja\n\n            {% set secret_key = data.get(\'headers\', {}).get(\'X-My-Secret-Key\') %}\n            {% set build = data.get(\'post\', {}) %}\n\n            {% if secret_key == \'F0fAgoQjIT@W\' and build.result == True %}\n            deploy_my_app:\n              cmd.state.sls:\n                - tgt: \'application*\'\n                - arg:\n                  - myapp.deploy\n                - kwarg:\n                    pillar:\n                      revision: {{ revision }}\n            {% endif %}\n        '
        disable_auth = self.application.mod_opts.get('webhook_disable_auth')
        if not disable_auth and (not self._verify_auth()):
            self.redirect('/login')
            return
        tag = 'salt/netapi/hook'
        if tag_suffix:
            tag += tag_suffix
        self.event = salt.utils.event.get_event('master', self.application.opts['sock_dir'], opts=self.application.opts, listen=False)
        arguments = {}
        for argname in self.request.query_arguments:
            value = self.get_arguments(argname)
            if len(value) == 1:
                value = value[0]
            arguments[argname] = value
        ret = self.event.fire_event({'post': self.request_payload, 'get': arguments, 'headers': dict(self.request.headers)}, tag)
        self.write(self.serialize({'success': ret}))
        self.finish()

def _check_cors_origin(origin, allowed_origins):
    if False:
        i = 10
        return i + 15
    '\n    Check if an origin match cors allowed origins\n    '
    if isinstance(allowed_origins, list):
        if origin in allowed_origins:
            return origin
    elif allowed_origins == '*':
        return allowed_origins
    elif allowed_origins == origin:
        return allowed_origins