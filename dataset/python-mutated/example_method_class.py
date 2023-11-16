class PetsView:
    mycontent = 'demonstrate return from MethodView class'

    def get(self, **kwargs):
        if False:
            i = 10
            return i + 15
        if kwargs:
            kwargs.update({'name': 'get'})
            return kwargs
        else:
            return [{'name': 'get'}]

    def search(self):
        if False:
            for i in range(10):
                print('nop')
        return [{'name': 'search'}]

    def post(self, **kwargs):
        if False:
            print('Hello World!')
        kwargs.update({'name': 'post'})
        return (kwargs, 201)

    def put(self, *args, **kwargs):
        if False:
            return 10
        kwargs.update({'name': 'put'})
        return (kwargs, 201)

    def delete(self, **kwargs):
        if False:
            return 10
        return 201

    def api_list(self):
        if False:
            i = 10
            return i + 15
        return 'api_list'

    def post_greeting(self):
        if False:
            return 10
        return 'post_greeting'