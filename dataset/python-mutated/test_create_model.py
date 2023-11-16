import platform
from typing import Generic, Optional, Tuple, TypeVar
import pytest
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, PydanticDeprecatedSince20, PydanticUserError, ValidationError, create_model, errors, field_validator, validator
from pydantic.fields import ModelPrivateAttr

def test_create_model():
    if False:
        for i in range(10):
            print('nop')
    model = create_model('FooModel', foo=(str, ...), bar=(int, 123))
    assert issubclass(model, BaseModel)
    assert model.model_config == BaseModel.model_config
    assert model.__name__ == 'FooModel'
    assert model.model_fields.keys() == {'foo', 'bar'}
    assert not model.__pydantic_decorators__.validators
    assert not model.__pydantic_decorators__.root_validators
    assert not model.__pydantic_decorators__.field_validators
    assert not model.__pydantic_decorators__.field_serializers
    assert model.__module__ == 'tests.test_create_model'

def test_create_model_usage():
    if False:
        while True:
            i = 10
    model = create_model('FooModel', foo=(str, ...), bar=(int, 123))
    m = model(foo='hello')
    assert m.foo == 'hello'
    assert m.bar == 123
    with pytest.raises(ValidationError):
        model()
    with pytest.raises(ValidationError):
        model(foo='hello', bar='xxx')

def test_create_model_pickle(create_module):
    if False:
        while True:
            i = 10
    "\n    Pickle will work for dynamically created model only if it was defined globally with its class name\n    and module where it's defined was specified\n    "

    @create_module
    def module():
        if False:
            i = 10
            return i + 15
        import pickle
        from pydantic import create_model
        FooModel = create_model('FooModel', foo=(str, ...), bar=(int, 123), __module__=__name__)
        m = FooModel(foo='hello')
        d = pickle.dumps(m)
        m2 = pickle.loads(d)
        assert m2.foo == m.foo == 'hello'
        assert m2.bar == m.bar == 123
        assert m2 == m
        assert m2 is not m

def test_create_model_multi_inheritance():
    if False:
        for i in range(10):
            print('nop')

    class Mixin:
        pass
    Generic_T = Generic[TypeVar('T')]
    FooModel = create_model('FooModel', value=(int, ...), __base__=(BaseModel, Generic_T))
    assert FooModel.__orig_bases__ == (BaseModel, Generic_T)

def test_invalid_name():
    if False:
        i = 10
        return i + 15
    with pytest.warns(RuntimeWarning):
        model = create_model('FooModel', _foo=(str, ...))
    assert len(model.model_fields) == 0

def test_field_wrong_tuple():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(errors.PydanticUserError):
        create_model('FooModel', foo=(1, 2, 3))

def test_config_and_base():
    if False:
        i = 10
        return i + 15
    with pytest.raises(errors.PydanticUserError):
        create_model('FooModel', __config__=BaseModel.model_config, __base__=BaseModel)

def test_inheritance():
    if False:
        return 10

    class BarModel(BaseModel):
        x: int = 1
        y: int = 2
    model = create_model('FooModel', foo=(str, ...), bar=(int, 123), __base__=BarModel)
    assert model.model_fields.keys() == {'foo', 'bar', 'x', 'y'}
    m = model(foo='a', x=4)
    assert m.model_dump() == {'bar': 123, 'foo': 'a', 'x': 4, 'y': 2}
    model = create_model('FooModel', foo=(str, ...), bar=(int, 123), __base__=(BarModel,))
    assert model.model_fields.keys() == {'foo', 'bar', 'x', 'y'}
    m = model(foo='a', x=4)
    assert m.model_dump() == {'bar': 123, 'foo': 'a', 'x': 4, 'y': 2}

def test_custom_config():
    if False:
        i = 10
        return i + 15
    config = ConfigDict(frozen=True)
    expected_config = BaseModel.model_config.copy()
    expected_config['frozen'] = True
    model = create_model('FooModel', foo=(int, ...), __config__=config)
    m = model(**{'foo': '987'})
    assert m.foo == 987
    assert model.model_config == expected_config
    with pytest.raises(ValidationError):
        m.foo = 654

def test_custom_config_inherits():
    if False:
        while True:
            i = 10

    class Config(ConfigDict):
        custom_config: bool
    config = Config(custom_config=True, validate_assignment=True)
    expected_config = Config(BaseModel.model_config)
    expected_config.update(config)
    model = create_model('FooModel', foo=(int, ...), __config__=config)
    m = model(**{'foo': '987'})
    assert m.foo == 987
    assert model.model_config == expected_config
    with pytest.raises(ValidationError):
        m.foo = ['123']

def test_custom_config_extras():
    if False:
        i = 10
        return i + 15
    config = ConfigDict(extra='forbid')
    model = create_model('FooModel', foo=(int, ...), __config__=config)
    assert model(foo=654)
    with pytest.raises(ValidationError):
        model(bar=654)

def test_inheritance_validators():
    if False:
        while True:
            i = 10

    class BarModel(BaseModel):

        @field_validator('a', check_fields=False)
        @classmethod
        def check_a(cls, v):
            if False:
                i = 10
                return i + 15
            if 'foobar' not in v:
                raise ValueError('"foobar" not found in a')
            return v
    model = create_model('FooModel', a=(str, 'cake'), __base__=BarModel)
    assert model().a == 'cake'
    assert model(a='this is foobar good').a == 'this is foobar good'
    with pytest.raises(ValidationError):
        model(a='something else')

def test_inheritance_validators_always():
    if False:
        while True:
            i = 10

    class BarModel(BaseModel):

        @field_validator('a', check_fields=False)
        @classmethod
        def check_a(cls, v):
            if False:
                for i in range(10):
                    print('nop')
            if 'foobar' not in v:
                raise ValueError('"foobar" not found in a')
            return v
    model = create_model('FooModel', a=(str, Field('cake', validate_default=True)), __base__=BarModel)
    with pytest.raises(ValidationError):
        model()
    assert model(a='this is foobar good').a == 'this is foobar good'
    with pytest.raises(ValidationError):
        model(a='something else')

def test_inheritance_validators_all():
    if False:
        for i in range(10):
            print('nop')
    with pytest.warns(PydanticDeprecatedSince20, match='Pydantic V1 style `@validator` validators are deprecated'):

        class BarModel(BaseModel):

            @validator('*')
            @classmethod
            def check_all(cls, v):
                if False:
                    return 10
                return v * 2
    model = create_model('FooModel', a=(int, ...), b=(int, ...), __base__=BarModel)
    assert model(a=2, b=6).model_dump() == {'a': 4, 'b': 12}

def test_funky_name():
    if False:
        for i in range(10):
            print('nop')
    model = create_model('FooModel', **{'this-is-funky': (int, ...)})
    m = model(**{'this-is-funky': '123'})
    assert m.model_dump() == {'this-is-funky': 123}
    with pytest.raises(ValidationError) as exc_info:
        model()
    assert exc_info.value.errors(include_url=False) == [{'input': {}, 'loc': ('this-is-funky',), 'msg': 'Field required', 'type': 'missing'}]

def test_repeat_base_usage():
    if False:
        return 10

    class Model(BaseModel):
        a: str
    assert Model.model_fields.keys() == {'a'}
    model = create_model('FooModel', b=(int, 1), __base__=Model)
    assert Model.model_fields.keys() == {'a'}
    assert model.model_fields.keys() == {'a', 'b'}
    model2 = create_model('Foo2Model', c=(int, 1), __base__=Model)
    assert Model.model_fields.keys() == {'a'}
    assert model.model_fields.keys() == {'a', 'b'}
    assert model2.model_fields.keys() == {'a', 'c'}
    model3 = create_model('Foo2Model', d=(int, 1), __base__=model)
    assert Model.model_fields.keys() == {'a'}
    assert model.model_fields.keys() == {'a', 'b'}
    assert model2.model_fields.keys() == {'a', 'c'}
    assert model3.model_fields.keys() == {'a', 'b', 'd'}

def test_dynamic_and_static():
    if False:
        for i in range(10):
            print('nop')

    class A(BaseModel):
        x: int
        y: float
        z: str
    DynamicA = create_model('A', x=(int, ...), y=(float, ...), z=(str, ...))
    for field_name in ('x', 'y', 'z'):
        assert A.model_fields[field_name].default == DynamicA.model_fields[field_name].default

def test_create_model_field_and_model_title():
    if False:
        while True:
            i = 10
    m = create_model('M', __config__=ConfigDict(title='abc'), a=(str, Field(title='field-title')))
    assert m.model_json_schema() == {'properties': {'a': {'title': 'field-title', 'type': 'string'}}, 'required': ['a'], 'title': 'abc', 'type': 'object'}

def test_create_model_field_description():
    if False:
        while True:
            i = 10
    m = create_model('M', a=(str, Field(description='descr')), __doc__='Some doc')
    assert m.model_json_schema() == {'properties': {'a': {'description': 'descr', 'title': 'A', 'type': 'string'}}, 'required': ['a'], 'title': 'M', 'type': 'object', 'description': 'Some doc'}

def test_create_model_with_doc():
    if False:
        print('Hello World!')
    model = create_model('FooModel', foo=(str, ...), bar=(int, 123), __doc__='The Foo model')
    assert model.__name__ == 'FooModel'
    assert model.__doc__ == 'The Foo model'

@pytest.mark.parametrize('base', [ModelPrivateAttr, object])
@pytest.mark.parametrize('use_annotation', [True, False])
def test_private_descriptors(base, use_annotation):
    if False:
        return 10
    set_name_calls = []
    get_calls = []
    set_calls = []
    delete_calls = []

    class MyDescriptor(base):

        def __init__(self, fn):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.fn = fn
            self.name = ''

        def __set_name__(self, owner, name):
            if False:
                i = 10
                return i + 15
            set_name_calls.append((owner, name))
            self.name = name

        def __get__(self, obj, type=None):
            if False:
                return 10
            get_calls.append((obj, type))
            return self.fn(obj) if obj else self

        def __set__(self, obj, value):
            if False:
                while True:
                    i = 10
            set_calls.append((obj, value))
            self.fn = lambda obj: value

        def __delete__(self, obj):
            if False:
                while True:
                    i = 10
            delete_calls.append(obj)

            def fail(obj):
                if False:
                    print('Hello World!')
                raise AttributeError(f'{self.name!r} is not defined on {obj!r}')
            self.fn = fail

    class A(BaseModel):
        x: int
        if use_annotation:
            _some_func: MyDescriptor = MyDescriptor(lambda self: self.x)
        else:
            _some_func = MyDescriptor(lambda self: self.x)

        @property
        def _double_x(self):
            if False:
                while True:
                    i = 10
            return self.x * 2
    assert set(A.__private_attributes__) == {'_some_func'}
    assert set_name_calls == [(A, '_some_func')]
    a = A(x=2)
    assert a._double_x == 4
    assert get_calls == []
    assert a._some_func == 2
    assert get_calls == [(a, A)]
    assert set_calls == []
    a._some_func = 3
    assert set_calls == [(a, 3)]
    assert a._some_func == 3
    assert get_calls == [(a, A), (a, A)]
    assert delete_calls == []
    del a._some_func
    assert delete_calls == [a]
    with pytest.raises(AttributeError, match="'_some_func' is not defined on A\\(x=2\\)"):
        a._some_func
    assert get_calls == [(a, A), (a, A), (a, A)]

def test_private_attr_set_name():
    if False:
        return 10

    class SetNameInt(int):
        _owner_attr_name: Optional[str] = None

        def __set_name__(self, owner, name):
            if False:
                print('Hello World!')
            self._owner_attr_name = f'{owner.__name__}.{name}'
    _private_attr_default = SetNameInt(1)

    class Model(BaseModel):
        _private_attr_1: int = PrivateAttr(default=_private_attr_default)
        _private_attr_2: SetNameInt = SetNameInt(2)
    assert _private_attr_default._owner_attr_name == 'Model._private_attr_1'
    m = Model()
    assert m._private_attr_1 == 1
    assert m._private_attr_1._owner_attr_name == 'Model._private_attr_1'
    assert m._private_attr_2 == 2
    assert m._private_attr_2._owner_attr_name == 'Model._private_attr_2'

def test_private_attr_default_descriptor_attribute_error():
    if False:
        return 10

    class SetNameInt(int):

        def __get__(self, obj, cls):
            if False:
                return 10
            return self
    _private_attr_default = SetNameInt(1)

    class Model(BaseModel):
        _private_attr: int = PrivateAttr(default=_private_attr_default)
    assert Model.__private_attributes__['_private_attr'].__get__(None, Model) == _private_attr_default
    with pytest.raises(AttributeError, match="'ModelPrivateAttr' object has no attribute 'some_attr'"):
        Model.__private_attributes__['_private_attr'].some_attr

def test_private_attr_set_name_do_not_crash_if_not_callable():
    if False:
        i = 10
        return i + 15

    class SetNameInt(int):
        __set_name__ = None
    _private_attr_default = SetNameInt(2)

    class Model(BaseModel):
        _private_attr: int = PrivateAttr(default=_private_attr_default)
    assert Model()._private_attr == 2

def test_del_model_attr():
    if False:
        i = 10
        return i + 15

    class Model(BaseModel):
        some_field: str
    m = Model(some_field='value')
    assert hasattr(m, 'some_field')
    del m.some_field
    assert not hasattr(m, 'some_field')

@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='In this single case `del` behaves weird on pypy')
def test_del_model_attr_error():
    if False:
        while True:
            i = 10

    class Model(BaseModel):
        some_field: str
    m = Model(some_field='value')
    assert not hasattr(m, 'other_field')
    with pytest.raises(AttributeError, match='other_field'):
        del m.other_field

def test_del_model_attr_with_privat_attrs():
    if False:
        for i in range(10):
            print('nop')

    class Model(BaseModel):
        _private_attr: int = PrivateAttr(default=1)
        some_field: str
    m = Model(some_field='value')
    assert hasattr(m, 'some_field')
    del m.some_field
    assert not hasattr(m, 'some_field')

@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='In this single case `del` behaves weird on pypy')
def test_del_model_attr_with_privat_attrs_error():
    if False:
        i = 10
        return i + 15

    class Model(BaseModel):
        _private_attr: int = PrivateAttr(default=1)
        some_field: str
    m = Model(some_field='value')
    assert not hasattr(m, 'other_field')
    with pytest.raises(AttributeError, match="'Model' object has no attribute 'other_field'"):
        del m.other_field

def test_del_model_attr_with_privat_attrs_twice_error():
    if False:
        while True:
            i = 10

    class Model(BaseModel):
        _private_attr: int = 1
        some_field: str
    m = Model(some_field='value')
    assert hasattr(m, '_private_attr')
    del m._private_attr
    with pytest.raises(AttributeError, match="'Model' object has no attribute '_private_attr'"):
        del m._private_attr

def test_create_model_with_slots():
    if False:
        print('Hello World!')
    field_definitions = {'__slots__': (Optional[Tuple[str, ...]], None), 'foobar': (Optional[int], None)}
    with pytest.warns(RuntimeWarning, match='__slots__ should not be passed to create_model'):
        model = create_model('PartialPet', **field_definitions)
    assert model.model_fields.keys() == {'foobar'}

def test_create_model_non_annotated():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError, match='A non-annotated attribute was detected: `bar = 123`. All model fields require a type annotation'):
        create_model('FooModel', foo=(str, ...), bar=123)

def test_create_model_tuple():
    if False:
        return 10
    model = create_model('FooModel', foo=(Tuple[int, int], (1, 2)))
    assert model().foo == (1, 2)
    assert model(foo=(3, 4)).foo == (3, 4)

def test_create_model_tuple_3():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(PydanticUserError, match='^Field definitions should be a `\\(<type>, <default>\\)`\\.\\n'):
        create_model('FooModel', foo=(Tuple[int, int], (1, 2), 'more'))

def test_create_model_protected_namespace_default():
    if False:
        i = 10
        return i + 15
    with pytest.warns(UserWarning, match='Field "model_prefixed_field" has conflict with protected namespace "model_"'):
        create_model('Model', model_prefixed_field=(str, ...))

def test_create_model_protected_namespace_real_conflict():
    if False:
        return 10
    with pytest.raises(NameError, match='Field "model_dump" conflicts with member .* of protected namespace "model_"'):
        create_model('Model', model_dump=(str, ...))

def test_create_model_custom_protected_namespace():
    if False:
        return 10
    with pytest.warns(UserWarning, match='Field "test_field" has conflict with protected namespace "test_"'):
        create_model('Model', __config__=ConfigDict(protected_namespaces=('test_',)), model_prefixed_field=(str, ...), test_field=(str, ...))

def test_create_model_multiple_protected_namespace():
    if False:
        print('Hello World!')
    with pytest.warns(UserWarning, match='Field "also_protect_field" has conflict with protected namespace "also_protect_"'):
        create_model('Model', __config__=ConfigDict(protected_namespaces=('protect_me_', 'also_protect_')), also_protect_field=(str, ...))

def test_json_schema_with_inner_models_with_duplicate_names():
    if False:
        for i in range(10):
            print('nop')
    model_a = create_model('a', inner=(str, ...))
    model_b = create_model('a', outer=(model_a, ...))
    assert model_b.model_json_schema() == {'$defs': {'a': {'properties': {'inner': {'title': 'Inner', 'type': 'string'}}, 'required': ['inner'], 'title': 'a', 'type': 'object'}}, 'properties': {'outer': {'$ref': '#/$defs/a'}}, 'required': ['outer'], 'title': 'a', 'type': 'object'}