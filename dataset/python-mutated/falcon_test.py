import falcon

class Resource(object):

    def on_get(self, req, resp):
        if False:
            return 10
        resp.status = falcon.HTTP_200
        resp.content_type = 'text/plain'
        resp.body = 'Hello, world!'
app = falcon.API()
app.add_route('/text', Resource())