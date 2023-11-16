from falcon.routing import CompiledRouter

class MyRouter(CompiledRouter):
    pass

class MyResponder:

    def on_get(self, req, res):
        if False:
            i = 10
            return i + 15
        pass

    def on_post(self, req, res):
        if False:
            return 10
        pass

    def on_delete(self, req, res):
        if False:
            i = 10
            return i + 15
        pass

    def on_get_id(self, req, res, id):
        if False:
            i = 10
            return i + 15
        pass

    def on_put_id(self, req, res, id):
        if False:
            print('Hello World!')
        pass

    def on_delete_id(self, req, res, id):
        if False:
            while True:
                i = 10
        pass

class MyResponderAsync:

    async def on_get(self, req, res):
        pass

    async def on_post(self, req, res):
        pass

    async def on_delete(self, req, res):
        pass

    async def on_get_id(self, req, res, id):
        pass

    async def on_put_id(self, req, res, id):
        pass

    async def on_delete_id(self, req, res, id):
        pass

class OtherResponder:

    def on_post_id(self, *args):
        if False:
            return 10
        pass

class OtherResponderAsync:

    async def on_post_id(self, *args):
        pass

def sinkFn(*args):
    if False:
        while True:
            i = 10
    pass

class SinkClass:

    def __call__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        pass

def my_error_handler(req, resp, ex, params):
    if False:
        while True:
            i = 10
    pass

async def my_error_handler_async(req, resp, ex, params):
    pass

class MyMiddleware:

    def process_request(self, *args):
        if False:
            return 10
        pass

    def process_resource(self, *args):
        if False:
            while True:
                i = 10
        pass

    def process_response(self, *args):
        if False:
            while True:
                i = 10
        pass

class OtherMiddleware:

    def process_request(self, *args):
        if False:
            return 10
        pass

    def process_response(self, *args):
        if False:
            i = 10
            return i + 15
        pass

class MyMiddlewareAsync:

    async def process_request(self, *args):
        pass

    async def process_resource(self, *args):
        pass

    async def process_response(self, *args):
        pass

class OtherMiddlewareAsync:

    async def process_request(self, *args):
        pass

    async def process_response(self, *args):
        pass