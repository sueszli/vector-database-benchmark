import datetime
from connexion import NoContent
from flask.views import MethodView

def example_decorator(f):
    if False:
        print('Hello World!')
    '\n    the returned view from <class>.as_view can be decorated\n    the decorator is initialized exactly once per class\n    '

    def decorator(*args, **kwargs):
        if False:
            return 10
        return f(*args, **kwargs)
    return decorator

class PetsView(MethodView):
    """Create Pets service"""
    decorators = [example_decorator]
    pets = {}

    def __init__(self, pets=None):
        if False:
            print('Hello World!')
        if pets is not None:
            self.pets = pets

    def post(self, body: dict):
        if False:
            return 10
        name = body.get('name')
        tag = body.get('tag')
        count = len(self.pets)
        pet = {}
        pet['id'] = count + 1
        pet['tag'] = tag
        pet['name'] = name
        pet['last_updated'] = datetime.datetime.now()
        self.pets[pet['id']] = pet
        return (pet, 201)

    def put(self, petId, body: dict):
        if False:
            return 10
        name = body['name']
        tag = body.get('tag')
        pet = self.pets.get(petId, {'id': petId})
        pet['name'] = name
        pet['tag'] = tag
        pet['last_updated'] = datetime.datetime.now()
        self.pets[petId] = pet
        return (self.pets[petId], 201)

    def delete(self, petId):
        if False:
            print('Hello World!')
        id_ = int(petId)
        if self.pets.get(id_) is None:
            return (NoContent, 404)
        del self.pets[id_]
        return (NoContent, 204)

    def get(self, petId=None, limit=100):
        if False:
            i = 10
            return i + 15
        if petId is None:
            return list(self.pets.values())[0:limit]
        if self.pets.get(petId) is None:
            return (NoContent, 404)
        return self.pets[petId]