from graphene.test import Client
from ..data import setup
from ..schema import schema
setup()
client = Client(schema)

def test_str_schema(snapshot):
    if False:
        print('Hello World!')
    snapshot.assert_match(str(schema).strip())

def test_correctly_fetches_id_name_rebels(snapshot):
    if False:
        while True:
            i = 10
    query = '\n      query RebelsQuery {\n        rebels {\n          id\n          name\n        }\n      }\n    '
    snapshot.assert_match(client.execute(query))

def test_correctly_refetches_rebels(snapshot):
    if False:
        print('Hello World!')
    query = '\n      query RebelsRefetchQuery {\n        node(id: "RmFjdGlvbjox") {\n          id\n          ... on Faction {\n            name\n          }\n        }\n      }\n    '
    snapshot.assert_match(client.execute(query))

def test_correctly_fetches_id_name_empire(snapshot):
    if False:
        print('Hello World!')
    query = '\n      query EmpireQuery {\n        empire {\n          id\n          name\n        }\n      }\n    '
    snapshot.assert_match(client.execute(query))

def test_correctly_refetches_empire(snapshot):
    if False:
        for i in range(10):
            print('nop')
    query = '\n      query EmpireRefetchQuery {\n        node(id: "RmFjdGlvbjoy") {\n          id\n          ... on Faction {\n            name\n          }\n        }\n      }\n    '
    snapshot.assert_match(client.execute(query))

def test_correctly_refetches_xwing(snapshot):
    if False:
        print('Hello World!')
    query = '\n      query XWingRefetchQuery {\n        node(id: "U2hpcDox") {\n          id\n          ... on Ship {\n            name\n          }\n        }\n      }\n    '
    snapshot.assert_match(client.execute(query))