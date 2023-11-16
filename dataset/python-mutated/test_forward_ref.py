import dataclasses
import re
import sys
import typing
from typing import Any, Optional, Tuple
import pytest
from pydantic import BaseModel, PydanticUserError, ValidationError

def test_postponed_annotations(create_module):
    if False:
        return 10
    module = create_module('\nfrom __future__ import annotations\nfrom pydantic import BaseModel\n\nclass Model(BaseModel):\n    a: int\n')
    m = module.Model(a='123')
    assert m.model_dump() == {'a': 123}

def test_postponed_annotations_auto_model_rebuild(create_module):
    if False:
        print('Hello World!')
    module = create_module('\nfrom __future__ import annotations\nfrom pydantic import BaseModel\n\nclass Model(BaseModel):\n    a: Model\n')
    assert module.Model.model_fields['a'].annotation.__name__ == 'Model'

def test_forward_ref_auto_update_no_model(create_module):
    if False:
        print('Hello World!')

    @create_module
    def module():
        if False:
            for i in range(10):
                print('nop')
        from typing import Optional
        import pytest
        from pydantic import BaseModel, PydanticUserError

        class Foo(BaseModel):
            a: Optional['Bar'] = None
        with pytest.raises(PydanticUserError, match='`Foo` is not fully defined; you should define `Bar`,'):
            Foo(a={'b': {'a': {}}})

        class Bar(BaseModel):
            b: 'Foo'
    assert module.Bar.__pydantic_complete__ is True
    assert repr(module.Bar.model_fields['b']) == 'FieldInfo(annotation=Foo, required=True)'
    b = module.Bar(b={'a': {'b': {}}})
    assert b.model_dump() == {'b': {'a': {'b': {'a': None}}}}
    assert repr(module.Foo.model_fields['a']) == 'FieldInfo(annotation=Union[Bar, NoneType], required=False)'
    assert module.Foo.__pydantic_complete__ is False
    f = module.Foo(a={'b': {'a': {'b': {'a': None}}}})
    assert module.Foo.__pydantic_complete__ is True
    assert f.model_dump() == {'a': {'b': {'a': {'b': {'a': None}}}}}

def test_forward_ref_one_of_fields_not_defined(create_module):
    if False:
        i = 10
        return i + 15

    @create_module
    def module():
        if False:
            i = 10
            return i + 15
        from pydantic import BaseModel

        class Foo(BaseModel):
            foo: 'Foo'
            bar: 'Bar'
    assert {k: repr(v) for (k, v) in module.Foo.model_fields.items()} == {'foo': 'FieldInfo(annotation=Foo, required=True)', 'bar': "FieldInfo(annotation=ForwardRef('Bar'), required=True)"}

def test_basic_forward_ref(create_module):
    if False:
        while True:
            i = 10

    @create_module
    def module():
        if False:
            return 10
        from typing import ForwardRef, Optional
        from pydantic import BaseModel

        class Foo(BaseModel):
            a: int
        FooRef = ForwardRef('Foo')

        class Bar(BaseModel):
            b: Optional[FooRef] = None
    assert module.Bar().model_dump() == {'b': None}
    assert module.Bar(b={'a': '123'}).model_dump() == {'b': {'a': 123}}

def test_self_forward_ref_module(create_module):
    if False:
        while True:
            i = 10

    @create_module
    def module():
        if False:
            for i in range(10):
                print('nop')
        from typing import ForwardRef, Optional
        from pydantic import BaseModel
        FooRef = ForwardRef('Foo')

        class Foo(BaseModel):
            a: int = 123
            b: Optional[FooRef] = None
    assert module.Foo().model_dump() == {'a': 123, 'b': None}
    assert module.Foo(b={'a': '321'}).model_dump() == {'a': 123, 'b': {'a': 321, 'b': None}}

def test_self_forward_ref_collection(create_module):
    if False:
        while True:
            i = 10

    @create_module
    def module():
        if False:
            return 10
        from typing import Dict, List
        from pydantic import BaseModel

        class Foo(BaseModel):
            a: int = 123
            b: 'Foo' = None
            c: 'List[Foo]' = []
            d: 'Dict[str, Foo]' = {}
    assert module.Foo().model_dump() == {'a': 123, 'b': None, 'c': [], 'd': {}}
    assert module.Foo(b={'a': '321'}, c=[{'a': 234}], d={'bar': {'a': 345}}).model_dump() == {'a': 123, 'b': {'a': 321, 'b': None, 'c': [], 'd': {}}, 'c': [{'a': 234, 'b': None, 'c': [], 'd': {}}], 'd': {'bar': {'a': 345, 'b': None, 'c': [], 'd': {}}}}
    with pytest.raises(ValidationError) as exc_info:
        module.Foo(b={'a': '321'}, c=[{'b': 234}], d={'bar': {'a': 345}})
    assert exc_info.value.errors(include_url=False) == [{'type': 'model_type', 'loc': ('c', 0, 'b'), 'msg': 'Input should be a valid dictionary or instance of Foo', 'input': 234, 'ctx': {'class_name': 'Foo'}}]
    assert repr(module.Foo.model_fields['a']) == 'FieldInfo(annotation=int, required=False, default=123)'
    assert repr(module.Foo.model_fields['b']) == 'FieldInfo(annotation=Foo, required=False)'
    if sys.version_info < (3, 10):
        return
    assert repr(module.Foo.model_fields['c']) == 'FieldInfo(annotation=List[Foo], required=False, default=[])'
    assert repr(module.Foo.model_fields['d']) == 'FieldInfo(annotation=Dict[str, Foo], required=False, default={})'

def test_self_forward_ref_local(create_module):
    if False:
        print('Hello World!')

    @create_module
    def module():
        if False:
            print('Hello World!')
        from typing import ForwardRef
        from pydantic import BaseModel

        def main():
            if False:
                return 10
            Foo = ForwardRef('Foo')

            class Foo(BaseModel):
                a: int = 123
                b: Foo = None
            return Foo
    Foo = module.main()
    assert Foo().model_dump() == {'a': 123, 'b': None}
    assert Foo(b={'a': '321'}).model_dump() == {'a': 123, 'b': {'a': 321, 'b': None}}

def test_forward_ref_dataclass(create_module):
    if False:
        print('Hello World!')

    @create_module
    def module():
        if False:
            while True:
                i = 10
        from typing import Optional
        from pydantic.dataclasses import dataclass

        @dataclass
        class MyDataclass:
            a: int
            b: Optional['MyDataclass'] = None
    dc = module.MyDataclass(a=1, b={'a': 2, 'b': {'a': 3}})
    assert dataclasses.asdict(dc) == {'a': 1, 'b': {'a': 2, 'b': {'a': 3, 'b': None}}}

def test_forward_ref_sub_types(create_module):
    if False:
        print('Hello World!')

    @create_module
    def module():
        if False:
            for i in range(10):
                print('nop')
        from typing import ForwardRef, Union
        from pydantic import BaseModel

        class Leaf(BaseModel):
            a: str
        TreeType = Union[ForwardRef('Node'), Leaf]

        class Node(BaseModel):
            value: int
            left: TreeType
            right: TreeType
    Node = module.Node
    Leaf = module.Leaf
    data = {'value': 3, 'left': {'a': 'foo'}, 'right': {'value': 5, 'left': {'a': 'bar'}, 'right': {'a': 'buzz'}}}
    node = Node(**data)
    assert isinstance(node.left, Leaf)
    assert isinstance(node.right, Node)

def test_forward_ref_nested_sub_types(create_module):
    if False:
        for i in range(10):
            print('nop')

    @create_module
    def module():
        if False:
            print('Hello World!')
        from typing import ForwardRef, Tuple, Union
        from pydantic import BaseModel

        class Leaf(BaseModel):
            a: str
        TreeType = Union[Union[Tuple[ForwardRef('Node'), str], int], Leaf]

        class Node(BaseModel):
            value: int
            left: TreeType
            right: TreeType
    Node = module.Node
    Leaf = module.Leaf
    data = {'value': 3, 'left': {'a': 'foo'}, 'right': [{'value': 5, 'left': {'a': 'bar'}, 'right': {'a': 'buzz'}}, 'test']}
    node = Node(**data)
    assert isinstance(node.left, Leaf)
    assert isinstance(node.right[0], Node)

def test_self_reference_json_schema(create_module):
    if False:
        return 10

    @create_module
    def module():
        if False:
            i = 10
            return i + 15
        from typing import List
        from pydantic import BaseModel

        class Account(BaseModel):
            name: str
            subaccounts: List['Account'] = []
    Account = module.Account
    assert Account.model_json_schema() == {'allOf': [{'$ref': '#/$defs/Account'}], '$defs': {'Account': {'title': 'Account', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, 'subaccounts': {'title': 'Subaccounts', 'default': [], 'type': 'array', 'items': {'$ref': '#/$defs/Account'}}}, 'required': ['name']}}}

def test_self_reference_json_schema_with_future_annotations(create_module):
    if False:
        for i in range(10):
            print('nop')
    module = create_module('\nfrom __future__ import annotations\nfrom typing import List\nfrom pydantic import BaseModel\n\nclass Account(BaseModel):\n  name: str\n  subaccounts: List[Account] = []\n    ')
    Account = module.Account
    assert Account.model_json_schema() == {'allOf': [{'$ref': '#/$defs/Account'}], '$defs': {'Account': {'title': 'Account', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, 'subaccounts': {'title': 'Subaccounts', 'default': [], 'type': 'array', 'items': {'$ref': '#/$defs/Account'}}}, 'required': ['name']}}}

def test_circular_reference_json_schema(create_module):
    if False:
        i = 10
        return i + 15

    @create_module
    def module():
        if False:
            print('Hello World!')
        from typing import List
        from pydantic import BaseModel

        class Owner(BaseModel):
            account: 'Account'

        class Account(BaseModel):
            name: str
            owner: 'Owner'
            subaccounts: List['Account'] = []
    Account = module.Account
    assert Account.model_json_schema() == {'allOf': [{'$ref': '#/$defs/Account'}], '$defs': {'Account': {'title': 'Account', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, 'owner': {'$ref': '#/$defs/Owner'}, 'subaccounts': {'title': 'Subaccounts', 'default': [], 'type': 'array', 'items': {'$ref': '#/$defs/Account'}}}, 'required': ['name', 'owner']}, 'Owner': {'title': 'Owner', 'type': 'object', 'properties': {'account': {'$ref': '#/$defs/Account'}}, 'required': ['account']}}}

def test_circular_reference_json_schema_with_future_annotations(create_module):
    if False:
        while True:
            i = 10
    module = create_module('\nfrom __future__ import annotations\nfrom typing import List\nfrom pydantic import BaseModel\n\nclass Owner(BaseModel):\n  account: Account\n\nclass Account(BaseModel):\n  name: str\n  owner: Owner\n  subaccounts: List[Account] = []\n\n    ')
    Account = module.Account
    assert Account.model_json_schema() == {'allOf': [{'$ref': '#/$defs/Account'}], '$defs': {'Account': {'title': 'Account', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, 'owner': {'$ref': '#/$defs/Owner'}, 'subaccounts': {'title': 'Subaccounts', 'default': [], 'type': 'array', 'items': {'$ref': '#/$defs/Account'}}}, 'required': ['name', 'owner']}, 'Owner': {'title': 'Owner', 'type': 'object', 'properties': {'account': {'$ref': '#/$defs/Account'}}, 'required': ['account']}}}

def test_forward_ref_with_field(create_module):
    if False:
        i = 10
        return i + 15

    @create_module
    def module():
        if False:
            i = 10
            return i + 15
        from typing import ForwardRef, List
        import pytest
        from pydantic import BaseModel, Field
        Foo = ForwardRef('Foo')
        with pytest.raises(TypeError, match="The following constraints cannot be applied.*\\'gt\\'"):

            class Foo(BaseModel):
                c: List[Foo] = Field(..., gt=0)

def test_forward_ref_optional(create_module):
    if False:
        while True:
            i = 10
    module = create_module('\nfrom __future__ import annotations\nfrom pydantic import BaseModel, Field, ConfigDict\nfrom typing import List, Optional\n\n\nclass Spec(BaseModel):\n    spec_fields: List[str] = Field(..., alias="fields")\n    filter: Optional[str] = None\n    sort: Optional[str]\n\n\nclass PSpec(Spec):\n    g: Optional[GSpec] = None\n\n\nclass GSpec(Spec):\n    p: Optional[PSpec]\n\n# PSpec.model_rebuild()\n\nclass Filter(BaseModel):\n    g: Optional[GSpec] = None\n    p: Optional[PSpec]\n    ')
    Filter = module.Filter
    assert isinstance(Filter(p={'sort': 'some_field:asc', 'fields': []}), Filter)

def test_forward_ref_with_create_model(create_module):
    if False:
        i = 10
        return i + 15

    @create_module
    def module():
        if False:
            print('Hello World!')
        import pydantic
        Sub = pydantic.create_model('Sub', foo=(str, 'bar'), __module__=__name__)
        assert Sub
        Main = pydantic.create_model('Main', sub=('Sub', ...), __module__=__name__)
        instance = Main(sub={})
        assert instance.sub.model_dump() == {'foo': 'bar'}

def test_resolve_forward_ref_dataclass(create_module):
    if False:
        print('Hello World!')
    module = create_module('\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\n\nfrom pydantic import BaseModel\nfrom typing_extensions import Literal\n\n@dataclass\nclass Base:\n    literal: Literal[1, 2]\n\nclass What(BaseModel):\n    base: Base\n        ')
    m = module.What(base=module.Base(literal=1))
    assert m.base.literal == 1

def test_nested_forward_ref():
    if False:
        for i in range(10):
            print('nop')

    class NestedTuple(BaseModel):
        x: Tuple[int, Optional['NestedTuple']]
    obj = NestedTuple.model_validate({'x': ('1', {'x': ('2', {'x': ('3', None)})})})
    assert obj.model_dump() == {'x': (1, {'x': (2, {'x': (3, None)})})}

def test_discriminated_union_forward_ref(create_module):
    if False:
        while True:
            i = 10

    @create_module
    def module():
        if False:
            for i in range(10):
                print('nop')
        from typing import Union
        from typing_extensions import Literal
        from pydantic import BaseModel, Field

        class Pet(BaseModel):
            pet: Union['Cat', 'Dog'] = Field(discriminator='type')

        class Cat(BaseModel):
            type: Literal['cat']

        class Dog(BaseModel):
            type: Literal['dog']
    assert module.Pet.__pydantic_complete__ is False
    with pytest.raises(ValidationError, match="Input tag 'pika' found using 'type' does not match any of the expected tags: 'cat', 'dog'"):
        module.Pet.model_validate({'pet': {'type': 'pika'}})
    assert module.Pet.__pydantic_complete__ is True
    assert module.Pet.model_json_schema() == {'title': 'Pet', 'required': ['pet'], 'type': 'object', 'properties': {'pet': {'title': 'Pet', 'discriminator': {'mapping': {'cat': '#/$defs/Cat', 'dog': '#/$defs/Dog'}, 'propertyName': 'type'}, 'oneOf': [{'$ref': '#/$defs/Cat'}, {'$ref': '#/$defs/Dog'}]}}, '$defs': {'Cat': {'title': 'Cat', 'type': 'object', 'properties': {'type': {'const': 'cat', 'title': 'Type'}}, 'required': ['type']}, 'Dog': {'title': 'Dog', 'type': 'object', 'properties': {'type': {'const': 'dog', 'title': 'Type'}}, 'required': ['type']}}}

def test_class_var_as_string(create_module):
    if False:
        i = 10
        return i + 15
    module = create_module('\nfrom __future__ import annotations\nfrom typing import ClassVar\nfrom pydantic import BaseModel\n\nclass Model(BaseModel):\n    a: ClassVar[int]\n')
    assert module.Model.__class_vars__ == {'a'}

def test_json_encoder_str(create_module):
    if False:
        i = 10
        return i + 15
    module = create_module("\nfrom pydantic import BaseModel, ConfigDict, field_serializer\n\n\nclass User(BaseModel):\n    x: str\n\n\nFooUser = User\n\n\nclass User(BaseModel):\n    y: str\n\n\nclass Model(BaseModel):\n    foo_user: FooUser\n    user: User\n\n    @field_serializer('user')\n    def serialize_user(self, v):\n        return f'User({v.y})'\n\n")
    m = module.Model(foo_user={'x': 'user1'}, user={'y': 'user2'})
    assert m.model_dump_json() == '{"foo_user":{"x":"user1"},"user":"User(user2)"}'
skip_pep585 = pytest.mark.skipif(sys.version_info < (3, 9), reason='PEP585 generics only supported for python 3.9 and above')

@skip_pep585
def test_pep585_self_referencing_generics(create_module):
    if False:
        return 10
    module = create_module('\nfrom __future__ import annotations\nfrom pydantic import BaseModel\n\nclass SelfReferencing(BaseModel):\n    names: list[SelfReferencing]  # noqa: F821\n')
    SelfReferencing = module.SelfReferencing
    if sys.version_info >= (3, 10):
        assert repr(SelfReferencing.model_fields['names']) == 'FieldInfo(annotation=list[SelfReferencing], required=True)'
    obj = SelfReferencing(names=[SelfReferencing(names=[])])
    assert obj.names == [SelfReferencing(names=[])]

@skip_pep585
def test_pep585_recursive_generics(create_module):
    if False:
        for i in range(10):
            print('nop')

    @create_module
    def module():
        if False:
            return 10
        from typing import ForwardRef
        from pydantic import BaseModel
        HeroRef = ForwardRef('Hero')

        class Team(BaseModel):
            name: str
            heroes: list[HeroRef]

        class Hero(BaseModel):
            name: str
            teams: list[Team]
        Team.model_rebuild()
    assert repr(module.Team.model_fields['heroes']) == 'FieldInfo(annotation=list[Hero], required=True)'
    assert repr(module.Hero.model_fields['teams']) == 'FieldInfo(annotation=list[Team], required=True)'
    h = module.Hero(name='Ivan', teams=[module.Team(name='TheBest', heroes=[])])
    assert h.model_dump() == {'name': 'Ivan', 'teams': [{'name': 'TheBest', 'heroes': []}]}

@pytest.mark.skipif(sys.version_info < (3, 9), reason='needs 3.9 or newer')
def test_class_var_forward_ref(create_module):
    if False:
        while True:
            i = 10
    create_module('\nfrom __future__ import annotations\nfrom typing import ClassVar\nfrom pydantic import BaseModel\n\nclass WithClassVar(BaseModel):\n    Instances: ClassVar[dict[str, WithClassVar]] = {}\n')

def test_recursive_model(create_module):
    if False:
        while True:
            i = 10
    module = create_module('\nfrom __future__ import annotations\nfrom typing import Optional\nfrom pydantic import BaseModel\n\nclass Foobar(BaseModel):\n    x: int\n    y: Optional[Foobar] = None\n')
    f = module.Foobar(x=1, y={'x': 2})
    assert f.model_dump() == {'x': 1, 'y': {'x': 2, 'y': None}}
    assert f.model_fields_set == {'x', 'y'}
    assert f.y.model_fields_set == {'x'}

@pytest.mark.skipif(sys.version_info < (3, 10), reason='needs 3.10 or newer')
def test_recursive_models_union(create_module):
    if False:
        print('Hello World!')
    module = create_module('\nfrom __future__ import annotations\n\nfrom pydantic import BaseModel\nfrom typing import TypeVar, Generic\n\nT = TypeVar("T")\n\nclass Foo(BaseModel):\n    bar: Bar[str] | None = None\n    bar2: int | Bar[float]\n\nclass Bar(BaseModel, Generic[T]):\n    foo: Foo\n')
    assert module.Foo.model_fields['bar'].annotation == typing.Optional[module.Bar[str]]
    assert module.Foo.model_fields['bar2'].annotation == typing.Union[int, module.Bar[float]]
    assert module.Bar.model_fields['foo'].annotation == module.Foo

def test_force_rebuild():
    if False:
        print('Hello World!')

    class Foobar(BaseModel):
        b: int
    assert Foobar.__pydantic_complete__ is True
    assert Foobar.model_rebuild() is None
    assert Foobar.model_rebuild(force=True) is True

def test_rebuild_subclass_of_built_model():
    if False:
        for i in range(10):
            print('nop')

    class Model(BaseModel):
        x: int

    class FutureReferencingModel(Model):
        y: 'FutureModel'

    class FutureModel(BaseModel):
        pass
    FutureReferencingModel.model_rebuild()
    assert FutureReferencingModel(x=1, y=FutureModel()).model_dump() == {'x': 1, 'y': {}}

def test_nested_annotation(create_module):
    if False:
        i = 10
        return i + 15
    module = create_module('\nfrom __future__ import annotations\nfrom pydantic import BaseModel\n\ndef nested():\n    class Foo(BaseModel):\n        a: int\n\n    class Bar(BaseModel):\n        b: Foo\n\n    return Bar\n')
    bar_model = module.nested()
    assert bar_model.__pydantic_complete__ is True
    assert bar_model(b={'a': 1}).model_dump() == {'b': {'a': 1}}

def test_nested_more_annotation(create_module):
    if False:
        print('Hello World!')

    @create_module
    def module():
        if False:
            for i in range(10):
                print('nop')
        from pydantic import BaseModel

        def nested():
            if False:
                i = 10
                return i + 15

            class Foo(BaseModel):
                a: int

            def more_nested():
                if False:
                    i = 10
                    return i + 15

                class Bar(BaseModel):
                    b: 'Foo'
                return Bar
            return more_nested()
    bar_model = module.nested()
    assert bar_model.__pydantic_complete__ is False

def test_nested_annotation_priority(create_module):
    if False:
        print('Hello World!')

    @create_module
    def module():
        if False:
            print('Hello World!')
        from annotated_types import Gt
        from typing_extensions import Annotated
        from pydantic import BaseModel
        Foobar = Annotated[int, Gt(0)]

        def nested():
            if False:
                return 10
            Foobar = Annotated[int, Gt(10)]

            class Bar(BaseModel):
                b: 'Foobar'
            return Bar
    bar_model = module.nested()
    assert bar_model.model_fields['b'].metadata[0].gt == 10
    assert bar_model(b=11).model_dump() == {'b': 11}
    with pytest.raises(ValidationError, match='Input should be greater than 10 \\[type=greater_than,'):
        bar_model(b=1)

def test_nested_model_rebuild(create_module):
    if False:
        for i in range(10):
            print('nop')

    @create_module
    def module():
        if False:
            return 10
        from pydantic import BaseModel

        def nested():
            if False:
                i = 10
                return i + 15

            class Bar(BaseModel):
                b: 'Foo'

            class Foo(BaseModel):
                a: int
            assert Bar.__pydantic_complete__ is False
            Bar.model_rebuild()
            return Bar
    bar_model = module.nested()
    assert bar_model.__pydantic_complete__ is True
    assert bar_model(b={'a': 1}).model_dump() == {'b': {'a': 1}}

def pytest_raises_user_error_for_undefined_type(defining_class_name, missing_type_name):
    if False:
        while True:
            i = 10
    "\n    Returns a `pytest.raises` context manager that checks the error message when an undefined type is present.\n\n    usage: `with pytest_raises_user_error_for_undefined_type(class_name='Foobar', missing_class_name='UndefinedType'):`\n    "
    return pytest.raises(PydanticUserError, match=re.escape(f'`{defining_class_name}` is not fully defined; you should define `{missing_type_name}`, then call `{defining_class_name}.model_rebuild()`.'))

def test_undefined_types_warning_1a_raised_by_default_2a_future_annotations(create_module):
    if False:
        for i in range(10):
            print('nop')
    with pytest_raises_user_error_for_undefined_type(defining_class_name='Foobar', missing_type_name='UndefinedType'):
        create_module('\nfrom __future__ import annotations\nfrom pydantic import BaseModel\n\nclass Foobar(BaseModel):\n    a: UndefinedType\n\n# Trigger the error for an undefined type:\nFoobar(a=1)\n')

def test_undefined_types_warning_1a_raised_by_default_2b_forward_ref(create_module):
    if False:
        while True:
            i = 10
    with pytest_raises_user_error_for_undefined_type(defining_class_name='Foobar', missing_type_name='UndefinedType'):

        @create_module
        def module():
            if False:
                while True:
                    i = 10
            from typing import ForwardRef
            from pydantic import BaseModel
            UndefinedType = ForwardRef('UndefinedType')

            class Foobar(BaseModel):
                a: UndefinedType
            Foobar(a=1)

def test_undefined_types_warning_1b_suppressed_via_config_2a_future_annotations(create_module):
    if False:
        return 10
    module = create_module("\nfrom __future__ import annotations\nfrom pydantic import BaseModel\n\n# Because we don't instantiate the type, no error for an undefined type is raised\nclass Foobar(BaseModel):\n    a: UndefinedType\n")
    assert module.Foobar.__pydantic_complete__ is False

def test_undefined_types_warning_1b_suppressed_via_config_2b_forward_ref(create_module):
    if False:
        print('Hello World!')

    @create_module
    def module():
        if False:
            while True:
                i = 10
        from typing import ForwardRef
        from pydantic import BaseModel
        UndefinedType = ForwardRef('UndefinedType')

        class Foobar(BaseModel):
            a: UndefinedType
    assert module.Foobar.__pydantic_complete__ is False

def test_undefined_types_warning_raised_by_usage(create_module):
    if False:
        return 10
    with pytest_raises_user_error_for_undefined_type('Foobar', 'UndefinedType'):

        @create_module
        def module():
            if False:
                i = 10
                return i + 15
            from typing import ForwardRef
            from pydantic import BaseModel
            UndefinedType = ForwardRef('UndefinedType')

            class Foobar(BaseModel):
                a: UndefinedType
            Foobar(a=1)

def test_rebuild_recursive_schema():
    if False:
        i = 10
        return i + 15
    from typing import ForwardRef, List

    class Expressions_(BaseModel):
        model_config = dict(undefined_types_warning=False)
        items: List["types['Expression']"]

    class Expression_(BaseModel):
        model_config = dict(undefined_types_warning=False)
        Or: ForwardRef("types['allOfExpressions']")
        Not: ForwardRef("types['allOfExpression']")

    class allOfExpression_(BaseModel):
        model_config = dict(undefined_types_warning=False)
        Not: ForwardRef("types['Expression']")

    class allOfExpressions_(BaseModel):
        model_config = dict(undefined_types_warning=False)
        items: List["types['Expression']"]
    types_namespace = {'types': {'Expression': Expression_, 'Expressions': Expressions_, 'allOfExpression': allOfExpression_, 'allOfExpressions': allOfExpressions_}}
    models = [allOfExpressions_, Expressions_]
    for m in models:
        m.model_rebuild(_types_namespace=types_namespace)

def test_forward_ref_in_generic(create_module: Any) -> None:
    if False:
        return 10
    'https://github.com/pydantic/pydantic/issues/6503'

    @create_module
    def module():
        if False:
            print('Hello World!')
        import typing as tp
        from pydantic import BaseModel

        class Foo(BaseModel):
            x: tp.Dict['tp.Type[Bar]', tp.Type['Bar']]

        class Bar(BaseModel):
            pass
    Foo = module.Foo
    Bar = module.Bar
    assert Foo(x={Bar: Bar}).x[Bar] is Bar

def test_forward_ref_in_generic_separate_modules(create_module: Any) -> None:
    if False:
        i = 10
        return i + 15
    'https://github.com/pydantic/pydantic/issues/6503'

    @create_module
    def module_1():
        if False:
            while True:
                i = 10
        import typing as tp
        from pydantic import BaseModel

        class Foo(BaseModel):
            x: tp.Dict['tp.Type[Bar]', tp.Type['Bar']]

    @create_module
    def module_2():
        if False:
            return 10
        from pydantic import BaseModel

        class Bar(BaseModel):
            pass
    Foo = module_1.Foo
    Bar = module_2.Bar
    Foo.model_rebuild(_types_namespace={'tp': typing, 'Bar': Bar})
    assert Foo(x={Bar: Bar}).x[Bar] is Bar