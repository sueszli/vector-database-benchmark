import falcon

class Things:

    def on_get(self, req, resp):
        if False:
            i = 10
            return i + 15
        pass
api = application = falcon.App()
api.add_route('/', Things())
if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    server = make_server('localhost', 8000, application)
    server.serve_forever()