from os import stat
from datasette import hookimpl, Response

@hookimpl(trylast=True)
def forbidden(datasette, request, message):
    if False:
        i = 10
        return i + 15

    async def inner():
        return Response.html(await datasette.render_template('error.html', {'title': 'Forbidden', 'error': message}, request=request), status=403)
    return inner