from litestar import Litestar, MediaType, Request, Response, get

@get('/resource', sync_to_thread=False)
def retrieve_resource(request: Request) -> Response[bytes]:
    if False:
        while True:
            i = 10
    provided_types = [MediaType.TEXT, MediaType.HTML, 'application/xml']
    preferred_type = request.accept.best_match(provided_types, default=MediaType.TEXT)
    content = None
    if preferred_type == MediaType.TEXT:
        content = b'Hello World!'
    elif preferred_type == MediaType.HTML:
        content = b'<h1>Hello World!</h1>'
    elif preferred_type == 'application/xml':
        content = b'<xml><msg>Hello World!</msg></xml>'
    return Response(content=content, media_type=preferred_type)
app = Litestar(route_handlers=[retrieve_resource])