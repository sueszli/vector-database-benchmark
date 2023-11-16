from sys import modules
from types import ModuleType
import strawberry
deferred_module_source = '\nfrom __future__ import annotations\n\nimport strawberry\n\n@strawberry.type\nclass User:\n    username: str\n    email: str\n\n@strawberry.interface\nclass UserContent:\n    created_by: User\n'

def test_deferred_other_module():
    if False:
        for i in range(10):
            print('nop')
    mod = ModuleType('tests.deferred_module')
    modules[mod.__name__] = mod
    try:
        exec(deferred_module_source, mod.__dict__)

        @strawberry.type
        class Post(mod.UserContent):
            title: str
            body: str
        definition = Post.__strawberry_definition__
        assert definition.fields[0].type == mod.User
    finally:
        del modules[mod.__name__]