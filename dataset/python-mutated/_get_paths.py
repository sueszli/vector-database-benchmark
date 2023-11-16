from ._utils import AttributeDict
from . import exceptions
CONFIG = AttributeDict()

def get_asset_url(path):
    if False:
        return 10
    return app_get_asset_url(CONFIG, path)

def app_get_asset_url(config, path):
    if False:
        while True:
            i = 10
    if config.assets_external_path:
        prefix = config.assets_external_path
    else:
        prefix = config.requests_pathname_prefix
    return '/'.join([prefix.rstrip('/'), config.assets_url_path.lstrip('/'), path])

def get_relative_path(path):
    if False:
        print('Hello World!')
    '\n    Return a path with `requests_pathname_prefix` prefixed before it.\n    Use this function when specifying local URL paths that will work\n    in environments regardless of what `requests_pathname_prefix` is.\n    In some deployment environments, like Dash Enterprise,\n    `requests_pathname_prefix` is set to the application name,\n    e.g. `my-dash-app`.\n    When working locally, `requests_pathname_prefix` might be unset and\n    so a relative URL like `/page-2` can just be `/page-2`.\n    However, when the app is deployed to a URL like `/my-dash-app`, then\n    `dash.get_relative_path(\'/page-2\')` will return `/my-dash-app/page-2`.\n    This can be used as an alternative to `get_asset_url` as well with\n    `dash.get_relative_path(\'/assets/logo.png\')`\n\n    Use this function with `dash.strip_relative_path` in callbacks that\n    deal with `dcc.Location` `pathname` routing.\n    That is, your usage may look like:\n    ```\n    app.layout = html.Div([\n        dcc.Location(id=\'url\'),\n        html.Div(id=\'content\')\n    ])\n    @dash.callback(Output(\'content\', \'children\'), [Input(\'url\', \'pathname\')])\n    def display_content(path):\n        page_name = dash.strip_relative_path(path)\n        if not page_name:  # None or \'\'\n            return html.Div([\n                dcc.Link(href=dash.get_relative_path(\'/page-1\')),\n                dcc.Link(href=dash.get_relative_path(\'/page-2\')),\n            ])\n        elif page_name == \'page-1\':\n            return chapters.page_1\n        if page_name == "page-2":\n            return chapters.page_2\n    ```\n    '
    return app_get_relative_path(CONFIG.requests_pathname_prefix, path)

def app_get_relative_path(requests_pathname, path):
    if False:
        while True:
            i = 10
    if requests_pathname == '/' and path == '':
        return '/'
    if requests_pathname != '/' and path == '':
        return requests_pathname
    if not path.startswith('/'):
        raise exceptions.UnsupportedRelativePath(f"\n            Paths that aren't prefixed with a leading / are not supported.\n            You supplied: {path}\n            ")
    return '/'.join([requests_pathname.rstrip('/'), path.lstrip('/')])

def strip_relative_path(path):
    if False:
        print('Hello World!')
    '\n    Return a path with `requests_pathname_prefix` and leading and trailing\n    slashes stripped from it. Also, if None is passed in, None is returned.\n    Use this function with `get_relative_path` in callbacks that deal\n    with `dcc.Location` `pathname` routing.\n    That is, your usage may look like:\n    ```\n    app.layout = html.Div([\n        dcc.Location(id=\'url\'),\n        html.Div(id=\'content\')\n    ])\n    @dash.callback(Output(\'content\', \'children\'), [Input(\'url\', \'pathname\')])\n    def display_content(path):\n        page_name = dash.strip_relative_path(path)\n        if not page_name:  # None or \'\'\n            return html.Div([\n                dcc.Link(href=dash.get_relative_path(\'/page-1\')),\n                dcc.Link(href=dash.get_relative_path(\'/page-2\')),\n            ])\n        elif page_name == \'page-1\':\n            return chapters.page_1\n        if page_name == "page-2":\n            return chapters.page_2\n    ```\n    Note that `chapters.page_1` will be served if the user visits `/page-1`\n    _or_ `/page-1/` since `strip_relative_path` removes the trailing slash.\n\n    Also note that `strip_relative_path` is compatible with\n    `get_relative_path` in environments where `requests_pathname_prefix` set.\n    In some deployment environments, like Dash Enterprise,\n    `requests_pathname_prefix` is set to the application name, e.g. `my-dash-app`.\n    When working locally, `requests_pathname_prefix` might be unset and\n    so a relative URL like `/page-2` can just be `/page-2`.\n    However, when the app is deployed to a URL like `/my-dash-app`, then\n    `dash.get_relative_path(\'/page-2\')` will return `/my-dash-app/page-2`\n\n    The `pathname` property of `dcc.Location` will return \'`/my-dash-app/page-2`\'\n    to the callback.\n    In this case, `dash.strip_relative_path(\'/my-dash-app/page-2\')`\n    will return `\'page-2\'`\n\n    For nested URLs, slashes are still included:\n    `dash.strip_relative_path(\'/page-1/sub-page-1/\')` will return\n    `page-1/sub-page-1`\n    ```\n    '
    return app_strip_relative_path(CONFIG.requests_pathname_prefix, path)

def app_strip_relative_path(requests_pathname, path):
    if False:
        return 10
    if path is None:
        return None
    if requests_pathname != '/' and (not path.startswith(requests_pathname.rstrip('/'))) or (requests_pathname == '/' and (not path.startswith('/'))):
        raise exceptions.UnsupportedRelativePath(f"\n            Paths that aren't prefixed with requests_pathname_prefix are not supported.\n            You supplied: {path} and requests_pathname_prefix was {requests_pathname}\n            ")
    if requests_pathname != '/' and path.startswith(requests_pathname.rstrip('/')):
        path = path.replace(requests_pathname.rstrip('/'), '', 1)
    return path.strip('/')