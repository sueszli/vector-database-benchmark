from flask.views import MethodView

class PetsView(MethodView):
    mycontent = 'demonstrate return from MethodView class'

    def get(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if kwargs:
            kwargs.update({'name': 'get'})
            return kwargs
        else:
            return [{'name': 'get'}]

    def search(self):
        if False:
            return 10
        return [{'name': 'search'}]

    def post(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs.update({'name': 'post'})
        return (kwargs, 201)

    def put(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        kwargs.update({'name': 'put'})
        return (kwargs, 201)

    def api_list(self):
        if False:
            while True:
                i = 10
        return 'api_list'

    def post_greeting(self):
        if False:
            for i in range(10):
                print('nop')
        return 'post_greeting'