from datasette import hookimpl
PERMITTED_VIEWS = {'table', 'query', 'database'}

@hookimpl
def extra_js_urls(view_name):
    if False:
        i = 10
        return i + 15
    print(view_name)
    if view_name in PERMITTED_VIEWS:
        return [{'url': f'/static/table-example-plugins.js'}]