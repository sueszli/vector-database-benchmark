from sanic import Sanic
from sanic.response import text
from sanic.views import HTTPMethodView
app = Sanic('some_name')

class SimpleView(HTTPMethodView):

    def get(self, request):
        if False:
            for i in range(10):
                print('nop')
        return text('I am get method')

    def post(self, request):
        if False:
            print('Hello World!')
        return text('I am post method')

    def put(self, request):
        if False:
            print('Hello World!')
        return text('I am put method')

    def patch(self, request):
        if False:
            i = 10
            return i + 15
        return text('I am patch method')

    def delete(self, request):
        if False:
            print('Hello World!')
        return text('I am delete method')

class SimpleAsyncView(HTTPMethodView):

    async def get(self, request):
        return text('I am async get method')

    async def post(self, request):
        return text('I am async post method')

    async def put(self, request):
        return text('I am async put method')
app.add_route(SimpleView.as_view(), '/')
app.add_route(SimpleAsyncView.as_view(), '/async')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)