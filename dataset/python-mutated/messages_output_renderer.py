from datasette import hookimpl

def render_message_debug(datasette, request):
    if False:
        while True:
            i = 10
    if request.args.get('add_msg'):
        msg_type = request.args.get('type', 'INFO')
        datasette.add_message(request, request.args['add_msg'], getattr(datasette, msg_type))
    return {'body': 'Hello from message debug'}

@hookimpl
def register_output_renderer(datasette):
    if False:
        i = 10
        return i + 15
    return [{'extension': 'message', 'render': render_message_debug, 'can_render': lambda : False}]