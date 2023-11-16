from datasette import hookimpl

@hookimpl
def extra_template_vars(view_name, request):
    if False:
        i = 10
        return i + 15
    return {'view_name': view_name, 'request': request}