"""
Plugin which tells Pylint how to handle mongoengine document classes.
"""
import astroid
from astroid import MANAGER
from astroid import nodes
CLASS_NAME_BLACKLIST = []

def register(linter):
    if False:
        print('Hello World!')
    pass

def transform(cls):
    if False:
        return 10
    if cls.name in CLASS_NAME_BLACKLIST:
        return
    if cls.name == 'StormFoundationDB':
        if '_fields' not in cls.locals:
            cls.locals['_fields'] = [nodes.Dict()]
    if cls.name.endswith('DB'):
        property_name = 'id'
        node = astroid.ClassDef(property_name, None)
        cls.locals[property_name] = [node]
MANAGER.register_transform(astroid.ClassDef, transform)