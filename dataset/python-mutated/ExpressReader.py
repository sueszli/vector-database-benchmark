"""Parse an EXPRESS file and extract basic information on all
entities and data types contained"""
import sys
import re
from collections import OrderedDict
re_match_entity = re.compile("\nENTITY\\s+(\\w+)\\s*                                    # 'ENTITY foo'\n.*?                                                  #  skip SUPERTYPE-of\n(?:SUBTYPE\\s+OF\\s+\\((\\w+)\\))?;                       # 'SUBTYPE OF (bar);' or simply ';'\n(.*?)                                                # 'a : atype;' (0 or more lines like this)\n(?:(?:INVERSE|UNIQUE|WHERE)\\s*$.*?)?                 #  skip the INVERSE, UNIQUE, WHERE clauses and everything behind \nEND_ENTITY;                                          \n", re.VERBOSE | re.DOTALL | re.MULTILINE)
re_match_type = re.compile('\nTYPE\\s+(\\w+?)\\s*=\\s*((?:LIST|SET)\\s*\\[\\d+:[\\d?]+\\]\\s*OF)?(?:\\s*UNIQUE)?\\s*(\\w+)   # TYPE foo = LIST[1:2] of blub\n(?:(?<=ENUMERATION)\\s*OF\\s*\\((.*?)\\))?\n.*?                                                                 #  skip the WHERE clause\nEND_TYPE;\n', re.VERBOSE | re.DOTALL)
re_match_field = re.compile('\n\\s+(\\w+?)\\s*:\\s*(OPTIONAL)?\\s*((?:LIST|SET)\\s*\\[\\d+:[\\d?]+\\]\\s*OF)?(?:\\s*UNIQUE)?\\s*(\\w+?);\n', re.VERBOSE | re.DOTALL)

class Schema:

    def __init__(self):
        if False:
            print('Hello World!')
        self.entities = OrderedDict()
        self.types = OrderedDict()

class Entity:

    def __init__(self, name, parent, members):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.parent = parent
        self.members = members

class Field:

    def __init__(self, name, type, optional, collection):
        if False:
            while True:
                i = 10
        self.name = name
        self.type = type
        self.optional = optional
        self.collection = collection
        self.fullspec = (self.collection + ' ' if self.collection else '') + self.type

class Type:

    def __init__(self, name, aggregate, equals, enums):
        if False:
            while True:
                i = 10
        self.name = name
        self.aggregate = aggregate
        self.equals = equals
        self.enums = enums

def read(filename, silent=False):
    if False:
        while True:
            i = 10
    schema = Schema()
    print('Try to read EXPRESS schema file' + filename)
    with open(filename, 'rt') as inp:
        contents = inp.read()
        types = re.findall(re_match_type, contents)
        for (name, aggregate, equals, enums) in types:
            schema.types[name] = Type(name, aggregate, equals, enums)
        entities = re.findall(re_match_entity, contents)
        for (name, parent, fields_raw) in entities:
            print('process entity {0}, parent is {1}'.format(name, parent)) if not silent else None
            fields = re.findall(re_match_field, fields_raw)
            members = [Field(name, type, opt, coll) for (name, opt, coll, type) in fields]
            print('  got {0} fields'.format(len(members))) if not silent else None
            schema.entities[name] = Entity(name, parent, members)
    return schema
if __name__ == '__main__':
    sys.exit(read(sys.argv[1] if len(sys.argv) > 1 else 'schema.exp'))