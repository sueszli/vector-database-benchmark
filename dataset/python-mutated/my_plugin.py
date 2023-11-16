import asyncio
from datasette import hookimpl, Permission
from datasette.facets import Facet
from datasette import tracer
from datasette.utils import path_with_added_args
from datasette.utils.asgi import asgi_send_json, Response
import base64
import pint
import json
ureg = pint.UnitRegistry()

@hookimpl
def prepare_connection(conn, database, datasette):
    if False:
        print('Hello World!')

    def convert_units(amount, from_, to_):
        if False:
            return 10
        "select convert_units(100, 'm', 'ft');"
        return (amount * ureg(from_)).to(to_).to_tuple()[0]
    conn.create_function('convert_units', 3, convert_units)

    def prepare_connection_args():
        if False:
            return 10
        return 'database={}, datasette.plugin_config("name-of-plugin")={}'.format(database, datasette.plugin_config('name-of-plugin'))
    conn.create_function('prepare_connection_args', 0, prepare_connection_args)

@hookimpl
def extra_css_urls(template, database, table, view_name, columns, request, datasette):
    if False:
        while True:
            i = 10

    async def inner():
        return ['https://plugin-example.datasette.io/{}/extra-css-urls-demo.css'.format(base64.b64encode(json.dumps({'template': template, 'database': database, 'table': table, 'view_name': view_name, 'request_path': request.path if request is not None else None, 'added': (await datasette.get_database().execute('select 3 * 5')).first()[0], 'columns': columns}).encode('utf8')).decode('utf8'))]
    return inner

@hookimpl
def extra_js_urls():
    if False:
        for i in range(10):
            print('nop')
    return [{'url': 'https://plugin-example.datasette.io/jquery.js', 'sri': 'SRIHASH'}, 'https://plugin-example.datasette.io/plugin1.js', {'url': 'https://plugin-example.datasette.io/plugin.module.js', 'module': True}]

@hookimpl
def extra_body_script(template, database, table, view_name, columns, request, datasette):
    if False:
        print('Hello World!')

    async def inner():
        script = 'var extra_body_script = {};'.format(json.dumps({'template': template, 'database': database, 'table': table, 'config': datasette.plugin_config('name-of-plugin', database=database, table=table), 'view_name': view_name, 'request_path': request.path if request is not None else None, 'added': (await datasette.get_database().execute('select 3 * 5')).first()[0], 'columns': columns}))
        return {'script': script, 'module': True}
    return inner

@hookimpl
def render_cell(row, value, column, table, database, datasette, request):
    if False:
        print('Hello World!')

    async def inner():
        if value == 'RENDER_CELL_DEMO':
            data = {'row': dict(row), 'column': column, 'table': table, 'database': database, 'config': datasette.plugin_config('name-of-plugin', database=database, table=table)}
            if request.args.get('_render_cell_extra'):
                data['render_cell_extra'] = 1
            return json.dumps(data)
        elif value == 'RENDER_CELL_ASYNC':
            return (await datasette.get_database(database).execute("select 'RENDER_CELL_ASYNC_RESULT'")).single_value()
    return inner

@hookimpl
def extra_template_vars(template, database, table, view_name, columns, request, datasette):
    if False:
        return 10
    return {'extra_template_vars': json.dumps({'template': template, 'scope_path': request.scope['path'] if request else None, 'columns': columns}, default=lambda b: b.decode('utf8'))}

@hookimpl
def prepare_jinja2_environment(env, datasette):
    if False:
        return 10

    async def select_times_three(s):
        db = datasette.get_database()
        return (await db.execute('select 3 * ?', [int(s)])).first()[0]

    async def inner():
        env.filters['select_times_three'] = select_times_three
    return inner

@hookimpl
def register_facet_classes():
    if False:
        for i in range(10):
            print('nop')
    return [DummyFacet]

class DummyFacet(Facet):
    type = 'dummy'

    async def suggest(self):
        columns = await self.get_columns(self.sql, self.params)
        return [{'name': column, 'toggle_url': self.ds.absolute_url(self.request, path_with_added_args(self.request, {'_facet_dummy': column})), 'type': 'dummy'} for column in columns] if self.request.args.get('_dummy_facet') else []

    async def facet_results(self):
        facet_results = {}
        facets_timed_out = []
        return (facet_results, facets_timed_out)

@hookimpl
def actor_from_request(datasette, request):
    if False:
        return 10
    if request.args.get('_bot'):
        return {'id': 'bot'}
    else:
        return None

@hookimpl
def asgi_wrapper():
    if False:
        i = 10
        return i + 15

    def wrap(app):
        if False:
            for i in range(10):
                print('nop')

        async def maybe_set_actor_in_scope(scope, receive, send):
            if b'_actor_in_scope' in scope.get('query_string', b''):
                scope = dict(scope, actor={'id': 'from-scope'})
                print(scope)
            await app(scope, receive, send)
        return maybe_set_actor_in_scope
    return wrap

@hookimpl
def permission_allowed(actor, action):
    if False:
        return 10
    if action == 'this_is_allowed':
        return True
    elif action == 'this_is_denied':
        return False
    elif action == 'view-database-download':
        return actor.get('can_download') if actor else None
    actor_id = None
    if actor:
        actor_id = actor.get('id')
    if actor_id == 'todomvc' and action in ('insert-row', 'create-table', 'drop-table', 'delete-row', 'update-row'):
        return True

@hookimpl
def register_routes():
    if False:
        for i in range(10):
            print('nop')

    async def one(datasette):
        return Response.text((await datasette.get_database().execute('select 1 + 1')).first()[0])

    async def two(request):
        name = request.url_vars['name']
        greeting = request.args.get('greeting')
        return Response.text(f'{greeting} {name}')

    async def three(scope, send):
        await asgi_send_json(send, {'hello': 'world'}, status=200, headers={'x-three': '1'})

    async def post(request):
        if request.method == 'GET':
            return Response.html(request.scope['csrftoken']())
        else:
            return Response.json(await request.post_vars())

    async def csrftoken_form(request, datasette):
        return Response.html(await datasette.render_template('csrftoken_form.html', request=request))

    def not_async():
        if False:
            i = 10
            return i + 15
        return Response.html('This was not async')

    def add_message(datasette, request):
        if False:
            while True:
                i = 10
        datasette.add_message(request, 'Hello from messages')
        return Response.html('Added message')

    async def render_message(datasette, request):
        return Response.html(await datasette.render_template('render_message.html', request=request))

    def login_as_root(datasette, request):
        if False:
            i = 10
            return i + 15
        if request.method == 'POST':
            response = Response.redirect('/')
            response.set_cookie('ds_actor', datasette.sign({'a': {'id': 'root'}}, 'actor'))
            return response
        return Response.html('\n            <form action="{}" method="POST">\n                <p>\n                    <input type="hidden" name="csrftoken" value="{}">\n                    <input type="submit"\n                      value="Sign in as root user"\n                      style="font-size: 2em; padding: 0.1em 0.5em;">\n                </p>\n            </form>\n        '.format(request.path, request.scope['csrftoken']()))

    def asgi_scope(scope):
        if False:
            print('Hello World!')
        return Response.json(scope, default=repr)

    async def parallel_queries(datasette):
        db = datasette.get_database()
        with tracer.trace_child_tasks():
            (one, two) = await asyncio.gather(db.execute('select coalesce(sleep(0.1), 1)'), db.execute('select coalesce(sleep(0.1), 2)'))
        return Response.json({'one': one.single_value(), 'two': two.single_value()})
    return [('/one/$', one), ('/two/(?P<name>.*)$', two), ('/three/$', three), ('/post/$', post), ('/csrftoken-form/$', csrftoken_form), ('/login-as-root$', login_as_root), ('/not-async/$', not_async), ('/add-message/$', add_message), ('/render-message/$', render_message), ('/asgi-scope$', asgi_scope), ('/parallel-queries$', parallel_queries)]

@hookimpl
def startup(datasette):
    if False:
        return 10
    datasette._startup_hook_fired = True
    from datasette import Response
    from datasette import Forbidden
    from datasette import NotFound
    from datasette import hookimpl
    from datasette import actor_matches_allow
    _ = (Response, Forbidden, NotFound, hookimpl, actor_matches_allow)

@hookimpl
def canned_queries(datasette, database, actor):
    if False:
        print('Hello World!')
    return {'from_hook': f"select 1, '{(actor['id'] if actor else 'null')}' as actor_id"}

@hookimpl
def register_magic_parameters():
    if False:
        i = 10
        return i + 15
    from uuid import uuid4

    def uuid(key, request):
        if False:
            while True:
                i = 10
        if key == 'new':
            return str(uuid4())
        else:
            raise KeyError

    def request(key, request):
        if False:
            for i in range(10):
                print('nop')
        if key == 'http_version':
            return request.scope['http_version']
        else:
            raise KeyError
    return [('request', request), ('uuid', uuid)]

@hookimpl
def forbidden(datasette, request, message):
    if False:
        i = 10
        return i + 15
    datasette._last_forbidden_message = message
    if request.path == '/data2':
        return Response.redirect('/login?message=' + message)

@hookimpl
def menu_links(datasette, actor, request):
    if False:
        i = 10
        return i + 15
    if actor:
        label = 'Hello'
        if request.args.get('_hello'):
            label += ', ' + request.args['_hello']
        return [{'href': datasette.urls.instance(), 'label': label}]

@hookimpl
def table_actions(datasette, database, table, actor):
    if False:
        print('Hello World!')
    if actor:
        return [{'href': datasette.urls.instance(), 'label': f'Database: {database}'}, {'href': datasette.urls.instance(), 'label': f'Table: {table}'}]

@hookimpl
def database_actions(datasette, database, actor, request):
    if False:
        return 10
    if actor:
        label = f'Database: {database}'
        if request.args.get('_hello'):
            label += ' - ' + request.args['_hello']
        return [{'href': datasette.urls.instance(), 'label': label}]

@hookimpl
def skip_csrf(scope):
    if False:
        for i in range(10):
            print('nop')
    return scope['path'] == '/skip-csrf'

@hookimpl
def register_permissions(datasette):
    if False:
        print('Hello World!')
    extras = datasette.plugin_config('datasette-register-permissions') or {}
    permissions = [Permission(name='permission-from-plugin', abbr='np', description='New permission added by a plugin', takes_database=True, takes_resource=False, default=False)]
    if extras:
        permissions.extend((Permission(name=p['name'], abbr=p['abbr'], description=p['description'], takes_database=p['takes_database'], takes_resource=p['takes_resource'], default=p['default']) for p in extras['permissions']))
    return permissions