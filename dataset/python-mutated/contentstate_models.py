import json
import random
import string
ALPHANUM = string.ascii_lowercase + string.digits

class Block:

    def __init__(self, typ, depth=0, key=None):
        if False:
            i = 10
            return i + 15
        self.type = typ
        self.depth = depth
        self.text = ''
        self.key = key if key else ''.join((random.choice(ALPHANUM) for _ in range(5)))
        self.inline_style_ranges = []
        self.entity_ranges = []

    def as_dict(self):
        if False:
            for i in range(10):
                print('nop')
        return {'key': self.key, 'type': self.type, 'depth': self.depth, 'text': self.text, 'inlineStyleRanges': [isr.as_dict() for isr in self.inline_style_ranges], 'entityRanges': [er.as_dict() for er in self.entity_ranges]}

class InlineStyleRange:

    def __init__(self, style):
        if False:
            for i in range(10):
                print('nop')
        self.style = style
        self.offset = None
        self.length = None

    def as_dict(self):
        if False:
            for i in range(10):
                print('nop')
        return {'offset': self.offset, 'length': self.length, 'style': self.style}

class Entity:

    def __init__(self, entity_type, mutability, data):
        if False:
            while True:
                i = 10
        self.entity_type = entity_type
        self.mutability = mutability
        self.data = data

    def as_dict(self):
        if False:
            i = 10
            return i + 15
        return {'mutability': self.mutability, 'type': self.entity_type, 'data': self.data}

class EntityRange:

    def __init__(self, key):
        if False:
            for i in range(10):
                print('nop')
        self.key = key
        self.offset = None
        self.length = None

    def as_dict(self):
        if False:
            print('Hello World!')
        return {'key': self.key, 'offset': self.offset, 'length': self.length}

class ContentState:
    """Pythonic representation of a Draftail contentState structure"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.blocks = []
        self.entity_count = 0
        self.entity_map = {}

    def add_entity(self, entity):
        if False:
            return 10
        key = self.entity_count
        self.entity_map[key] = entity
        self.entity_count += 1
        return key

    def as_dict(self):
        if False:
            print('Hello World!')
        return {'blocks': [block.as_dict() for block in self.blocks], 'entityMap': {key: entity.as_dict() for (key, entity) in self.entity_map.items()}}

    def as_json(self, **kwargs):
        if False:
            while True:
                i = 10
        return json.dumps(self.as_dict(), **kwargs)