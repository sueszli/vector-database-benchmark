from graphene.types.enum import Enum, EnumOptions
from graphene.types.inputobjecttype import InputObjectType
from graphene.types.objecttype import ObjectType, ObjectTypeOptions

class SpecialOptions(ObjectTypeOptions):
    other_attr = None

class SpecialObjectType(ObjectType):

    @classmethod
    def __init_subclass_with_meta__(cls, other_attr='default', **options):
        if False:
            i = 10
            return i + 15
        _meta = SpecialOptions(cls)
        _meta.other_attr = other_attr
        super(SpecialObjectType, cls).__init_subclass_with_meta__(_meta=_meta, **options)

def test_special_objecttype_could_be_subclassed():
    if False:
        print('Hello World!')

    class MyType(SpecialObjectType):

        class Meta:
            other_attr = 'yeah!'
    assert MyType._meta.other_attr == 'yeah!'

def test_special_objecttype_could_be_subclassed_default():
    if False:
        return 10

    class MyType(SpecialObjectType):
        pass
    assert MyType._meta.other_attr == 'default'

def test_special_objecttype_inherit_meta_options():
    if False:
        while True:
            i = 10

    class MyType(SpecialObjectType):
        pass
    assert MyType._meta.name == 'MyType'
    assert MyType._meta.default_resolver is None
    assert MyType._meta.interfaces == ()

class SpecialInputObjectTypeOptions(ObjectTypeOptions):
    other_attr = None

class SpecialInputObjectType(InputObjectType):

    @classmethod
    def __init_subclass_with_meta__(cls, other_attr='default', **options):
        if False:
            return 10
        _meta = SpecialInputObjectTypeOptions(cls)
        _meta.other_attr = other_attr
        super(SpecialInputObjectType, cls).__init_subclass_with_meta__(_meta=_meta, **options)

def test_special_inputobjecttype_could_be_subclassed():
    if False:
        while True:
            i = 10

    class MyInputObjectType(SpecialInputObjectType):

        class Meta:
            other_attr = 'yeah!'
    assert MyInputObjectType._meta.other_attr == 'yeah!'

def test_special_inputobjecttype_could_be_subclassed_default():
    if False:
        print('Hello World!')

    class MyInputObjectType(SpecialInputObjectType):
        pass
    assert MyInputObjectType._meta.other_attr == 'default'

def test_special_inputobjecttype_inherit_meta_options():
    if False:
        return 10

    class MyInputObjectType(SpecialInputObjectType):
        pass
    assert MyInputObjectType._meta.name == 'MyInputObjectType'

class SpecialEnumOptions(EnumOptions):
    other_attr = None

class SpecialEnum(Enum):

    @classmethod
    def __init_subclass_with_meta__(cls, other_attr='default', **options):
        if False:
            return 10
        _meta = SpecialEnumOptions(cls)
        _meta.other_attr = other_attr
        super(SpecialEnum, cls).__init_subclass_with_meta__(_meta=_meta, **options)

def test_special_enum_could_be_subclassed():
    if False:
        print('Hello World!')

    class MyEnum(SpecialEnum):

        class Meta:
            other_attr = 'yeah!'
    assert MyEnum._meta.other_attr == 'yeah!'

def test_special_enum_could_be_subclassed_default():
    if False:
        while True:
            i = 10

    class MyEnum(SpecialEnum):
        pass
    assert MyEnum._meta.other_attr == 'default'

def test_special_enum_inherit_meta_options():
    if False:
        while True:
            i = 10

    class MyEnum(SpecialEnum):
        pass
    assert MyEnum._meta.name == 'MyEnum'