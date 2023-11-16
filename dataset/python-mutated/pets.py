import datetime
from connexion import NoContent
pets = {}

def post(body):
    if False:
        for i in range(10):
            print('nop')
    name = body.get('name')
    tag = body.get('tag')
    count = len(pets)
    pet = {}
    pet['id'] = count + 1
    pet['tag'] = tag
    pet['name'] = name
    pet['last_updated'] = datetime.datetime.now()
    pets[pet['id']] = pet
    return (pet, 201)

def put(body):
    if False:
        i = 10
        return i + 15
    id_ = body['id']
    name = body['name']
    tag = body.get('tag')
    id_ = int(id_)
    pet = pets.get(id_, {'id': id_})
    pet['name'] = name
    pet['tag'] = tag
    pet['last_updated'] = datetime.datetime.now()
    pets[id_] = pet
    return pets[id_]

def delete(id_):
    if False:
        print('Hello World!')
    id_ = int(id_)
    if pets.get(id_) is None:
        return (NoContent, 404)
    del pets[id_]
    return (NoContent, 204)

def get(petId):
    if False:
        for i in range(10):
            print('nop')
    id_ = int(petId)
    if pets.get(id_) is None:
        return (NoContent, 404)
    return pets[id_]

def search(limit=100):
    if False:
        i = 10
        return i + 15
    return list(pets.values())[0:limit]