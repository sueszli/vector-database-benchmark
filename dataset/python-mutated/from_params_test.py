from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union
import pytest
import torch
from allennlp.common import Lazy, Params, Registrable
from allennlp.common.from_params import FromParams, takes_arg, remove_optional, create_kwargs
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import DataLoader, DatasetReader, Tokenizer
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.common.checks import ConfigurationError

class TestFromParams(AllenNlpTestCase):

    def test_takes_arg(self):
        if False:
            i = 10
            return i + 15

        def bare_function(some_input: int) -> int:
            if False:
                i = 10
                return i + 15
            return some_input + 1
        assert takes_arg(bare_function, 'some_input')
        assert not takes_arg(bare_function, 'some_other_input')

        class SomeClass:
            total = 0

            def __init__(self, constructor_param: str) -> None:
                if False:
                    return 10
                self.constructor_param = constructor_param

            def check_param(self, check: str) -> bool:
                if False:
                    i = 10
                    return i + 15
                return self.constructor_param == check

            @classmethod
            def set_total(cls, new_total: int) -> None:
                if False:
                    return 10
                cls.total = new_total
        assert takes_arg(SomeClass, 'self')
        assert takes_arg(SomeClass, 'constructor_param')
        assert not takes_arg(SomeClass, 'check')
        assert takes_arg(SomeClass.check_param, 'check')
        assert not takes_arg(SomeClass.check_param, 'other_check')
        assert takes_arg(SomeClass.set_total, 'new_total')
        assert not takes_arg(SomeClass.set_total, 'total')

    def test_remove_optional(self):
        if False:
            return 10
        optional_type = Optional[Dict[str, str]]
        bare_type = remove_optional(optional_type)
        bare_bare_type = remove_optional(bare_type)
        assert bare_type == Dict[str, str]
        assert bare_bare_type == Dict[str, str]
        assert remove_optional(Optional[str]) == str
        assert remove_optional(str) == str

    def test_from_params(self):
        if False:
            return 10
        my_class = MyClass.from_params(Params({'my_int': 10}), my_bool=True)
        assert isinstance(my_class, MyClass)
        assert my_class.my_int == 10
        assert my_class.my_bool

    def test_good_error_message_when_passing_non_params(self):
        if False:
            while True:
                i = 10
        from allennlp.nn import InitializerApplicator
        params = Params({'initializer': [['regex1', 'uniform'], ['regex2', 'orthogonal']]})
        with pytest.raises(ConfigurationError, match='dictionary.*InitializerApplicator'):
            InitializerApplicator.from_params(params=params.pop('initializer'))

    def test_create_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        kwargs = create_kwargs(MyClass, MyClass, Params({'my_int': 5}), my_bool=True, my_float=4.4)
        assert kwargs == {'my_int': 5, 'my_bool': True}

    def test_extras(self):
        if False:
            while True:
                i = 10
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size: int, name: str) -> None:
                if False:
                    i = 10
                    return i + 15
                self.size = size
                self.name = name

        @A.register('c')
        class C(A):

            def __init__(self, size: int, name: str) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.size = size
                self.name = name

            @classmethod
            def from_params(cls, params: Params, size: int, **extras) -> 'C':
                if False:
                    print('Hello World!')
                name = params.pop('name')
                return cls(size=size, name=name)
        params = Params({'type': 'b', 'size': 10})
        b = A.from_params(params, name='extra')
        assert b.name == 'extra'
        assert b.size == 10
        params = Params({'type': 'b', 'size': 10})
        b = A.from_params(params, name='extra', unwanted=True)
        assert b.name == 'extra'
        assert b.size == 10
        params = Params({'type': 'c', 'name': 'extra_c'})
        c = A.from_params(params, size=20)
        assert c.name == 'extra_c'
        assert c.size == 20
        params = Params({'type': 'c', 'name': 'extra_c'})
        c = A.from_params(params, size=20, unwanted=True)
        assert c.name == 'extra_c'
        assert c.size == 20

    def test_extras_for_custom_classes(self):
        if False:
            return 10
        from allennlp.common.registrable import Registrable

        class BaseClass(Registrable):
            pass

        class BaseClass2(Registrable):
            pass

        @BaseClass.register('A')
        class A(BaseClass):

            def __init__(self, a: int, b: int, val: str) -> None:
                if False:
                    print('Hello World!')
                self.a = a
                self.b = b
                self.val = val

            def __hash__(self):
                if False:
                    return 10
                return self.b

            def __eq__(self, other):
                if False:
                    print('Hello World!')
                return self.b == other.b

            @classmethod
            def from_params(cls, params: Params, a: int, **extras) -> 'A':
                if False:
                    while True:
                        i = 10
                b = params.pop_int('b')
                val = params.pop('val', 'C')
                params.assert_empty(cls.__name__)
                return cls(a=a, b=b, val=val)

        @BaseClass2.register('B')
        class B(BaseClass2):

            def __init__(self, c: int, b: int) -> None:
                if False:
                    i = 10
                    return i + 15
                self.c = c
                self.b = b

            @classmethod
            def from_params(cls, params: Params, c: int, **extras) -> 'B':
                if False:
                    i = 10
                    return i + 15
                b = params.pop_int('b')
                params.assert_empty(cls.__name__)
                return cls(c=c, b=b)

        @BaseClass.register('E')
        class E(BaseClass):

            def __init__(self, m: int, n: int) -> None:
                if False:
                    i = 10
                    return i + 15
                self.m = m
                self.n = n

            @classmethod
            def from_params(cls, params: Params, **extras2) -> 'E':
                if False:
                    return 10
                m = params.pop_int('m')
                params.assert_empty(cls.__name__)
                n = extras2['n']
                return cls(m=m, n=n)

        class C:
            pass

        @BaseClass.register('D')
        class D(BaseClass):

            def __init__(self, arg1: List[BaseClass], arg2: Tuple[BaseClass, BaseClass2], arg3: Dict[str, BaseClass], arg4: Set[BaseClass], arg5: List[BaseClass]) -> None:
                if False:
                    return 10
                self.arg1 = arg1
                self.arg2 = arg2
                self.arg3 = arg3
                self.arg4 = arg4
                self.arg5 = arg5
        vals = [1, 2, 3]
        params = Params({'type': 'D', 'arg1': [{'type': 'A', 'b': vals[0]}, {'type': 'A', 'b': vals[1]}, {'type': 'A', 'b': vals[2]}], 'arg2': [{'type': 'A', 'b': vals[0]}, {'type': 'B', 'b': vals[0]}], 'arg3': {'class_1': {'type': 'A', 'b': vals[0]}, 'class_2': {'type': 'A', 'b': vals[1]}}, 'arg4': [{'type': 'A', 'b': vals[0], 'val': 'M'}, {'type': 'A', 'b': vals[1], 'val': 'N'}, {'type': 'A', 'b': vals[1], 'val': 'N'}], 'arg5': [{'type': 'E', 'm': 9}]})
        extra = C()
        tval1 = 5
        tval2 = 6
        d = BaseClass.from_params(params=params, extra=extra, a=tval1, c=tval2, n=10)
        assert len(d.arg1) == len(vals)
        assert isinstance(d.arg1, list)
        assert isinstance(d.arg1[0], A)
        assert all((x.b == y for (x, y) in zip(d.arg1, vals)))
        assert all((x.a == tval1 for x in d.arg1))
        assert isinstance(d.arg2, tuple)
        assert isinstance(d.arg2[0], A)
        assert isinstance(d.arg2[1], B)
        assert d.arg2[0].a == tval1
        assert d.arg2[1].c == tval2
        assert d.arg2[0].b == d.arg2[1].b == vals[0]
        assert isinstance(d.arg3, dict)
        assert isinstance(d.arg3['class_1'], A)
        assert d.arg3['class_1'].a == d.arg3['class_2'].a == tval1
        assert d.arg3['class_1'].b == vals[0]
        assert d.arg3['class_2'].b == vals[1]
        assert isinstance(d.arg4, set)
        assert len(d.arg4) == 2
        assert any((x.val == 'M' for x in d.arg4))
        assert any((x.val == 'N' for x in d.arg4))
        assert isinstance(d.arg5, list)
        assert isinstance(d.arg5[0], E)
        assert d.arg5[0].m == 9
        assert d.arg5[0].n == 10

    def test_no_constructor(self):
        if False:
            while True:
                i = 10
        params = Params({'type': 'just_spaces'})
        Tokenizer.from_params(params)

    def test_union(self):
        if False:
            print('Hello World!')

        class A(FromParams):

            def __init__(self, a: Union[int, List[int]]) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.a = a

        class B(FromParams):

            def __init__(self, b: Union[A, List[A]]) -> None:
                if False:
                    while True:
                        i = 10
                self.b = b
        params = Params({'a': 3})
        a = A.from_params(params)
        assert a.a == 3
        params = Params({'a': [3, 4, 5]})
        a = A.from_params(params)
        assert a.a == [3, 4, 5]
        params = Params({'b': {'a': 3}})
        b = B.from_params(params)
        assert isinstance(b.b, A)
        assert b.b.a == 3
        params = Params({'b': [{'a': 3}, {'a': [4, 5]}]})
        b = B.from_params(params)
        assert isinstance(b.b, list)
        assert b.b[0].a == 3
        assert b.b[1].a == [4, 5]

    def test_crazy_nested_union(self):
        if False:
            while True:
                i = 10

        class A(FromParams):

            def __init__(self, a: Union[int, List[int]]) -> None:
                if False:
                    while True:
                        i = 10
                self.a = a

        class B(FromParams):

            def __init__(self, b: Union[A, List[A]]) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.b = b

        class C(FromParams):

            def __init__(self, c: Union[A, B, Dict[str, A]]) -> None:
                if False:
                    print('Hello World!')
                self.c = c
        params = Params({'c': {'a': {'a': 3}, 'b': {'a': [4, 5]}}})
        c = C.from_params(params)
        assert isinstance(c.c, dict)
        assert c.c['a'].a == 3
        assert c.c['b'].a == [4, 5]

    def test_union_of_castable_types(self):
        if False:
            print('Hello World!')

        class IntFloat(FromParams):

            def __init__(self, a: Union[int, float]) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.a = a

        class FloatInt(FromParams):

            def __init__(self, a: Union[float, int]) -> None:
                if False:
                    print('Hello World!')
                self.a = a
        float_param_str = '{"a": 1.0}'
        int_param_str = '{"a": 1}'
        import json
        for (expected_type, param_str) in [(int, int_param_str), (float, float_param_str)]:
            for cls in [IntFloat, FloatInt]:
                c = cls.from_params(Params(json.loads(param_str)))
                assert type(c.a) == expected_type

    def test_invalid_type_conversions(self):
        if False:
            print('Hello World!')

        class A(FromParams):

            def __init__(self, a: int) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.a = a
        with pytest.raises(TypeError):
            A.from_params(Params({'a': '1'}))
        with pytest.raises(TypeError):
            A.from_params(Params({'a': 1.0}))

    def test_dict(self):
        if False:
            return 10
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size: int) -> None:
                if False:
                    return 10
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items: Dict[str, A]) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.items = items
        params = Params({'type': 'd', 'items': {'first': {'type': 'b', 'size': 1}, 'second': {'type': 'b', 'size': 2}}})
        d = C.from_params(params)
        assert isinstance(d.items, dict)
        assert len(d.items) == 2
        assert all((isinstance(key, str) for key in d.items.keys()))
        assert all((isinstance(value, B) for value in d.items.values()))
        assert d.items['first'].size == 1
        assert d.items['second'].size == 2

    def test_dict_not_params(self):
        if False:
            print('Hello World!')

        class A(FromParams):

            def __init__(self, counts: Dict[str, int]) -> None:
                if False:
                    i = 10
                    return i + 15
                self.counts = counts
        params = Params({'counts': {'a': 10, 'b': 20}})
        a = A.from_params(params)
        assert isinstance(a.counts, dict)
        assert not isinstance(a.counts, Params)

    def test_list(self):
        if False:
            while True:
                i = 10
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size: int) -> None:
                if False:
                    return 10
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items: List[A]) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.items = items
        params = Params({'type': 'd', 'items': [{'type': 'b', 'size': 1}, {'type': 'b', 'size': 2}]})
        d = C.from_params(params)
        assert isinstance(d.items, list)
        assert len(d.items) == 2
        assert all((isinstance(item, B) for item in d.items))
        assert d.items[0].size == 1
        assert d.items[1].size == 2

    def test_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size: int) -> None:
                if False:
                    print('Hello World!')
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, name: str) -> None:
                if False:
                    print('Hello World!')
                self.name = name

        class E(Registrable):
            pass

        @E.register('f')
        class F(E):

            def __init__(self, items: Tuple[A, C]) -> None:
                if False:
                    print('Hello World!')
                self.items = items
        params = Params({'type': 'f', 'items': [{'type': 'b', 'size': 1}, {'type': 'd', 'name': 'item2'}]})
        f = E.from_params(params)
        assert isinstance(f.items, tuple)
        assert len(f.items) == 2
        assert isinstance(f.items[0], B)
        assert isinstance(f.items[1], D)
        assert f.items[0].size == 1
        assert f.items[1].name == 'item2'

    def test_set(self):
        if False:
            print('Hello World!')
        from allennlp.common.registrable import Registrable

        class A(Registrable):

            def __init__(self, name: str) -> None:
                if False:
                    print('Hello World!')
                self.name = name

            def __eq__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return self.name == other.name

            def __hash__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return hash(self.name)

        @A.register('b')
        class B(A):
            pass

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items: Set[A]) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.items = items
        params = Params({'type': 'd', 'items': [{'type': 'b', 'name': 'item1'}, {'type': 'b', 'name': 'item2'}, {'type': 'b', 'name': 'item2'}]})
        d = C.from_params(params)
        assert isinstance(d.items, set)
        assert len(d.items) == 2
        assert all((isinstance(item, B) for item in d.items))
        assert any((item.name == 'item1' for item in d.items))
        assert any((item.name == 'item2' for item in d.items))

    def test_transferring_of_modules(self):
        if False:
            for i in range(10):
                print('nop')
        model_archive = str(self.FIXTURES_ROOT / 'basic_classifier' / 'serialization' / 'model.tar.gz')
        trained_model = load_archive(model_archive).model
        config_file = str(self.FIXTURES_ROOT / 'basic_classifier' / 'experiment_seq2seq.jsonnet')
        model_params = Params.from_file(config_file).pop('model').as_dict(quiet=True)
        model_params['text_field_embedder'] = {'_pretrained': {'archive_file': model_archive, 'module_path': '_text_field_embedder', 'freeze': True}}
        model_params['seq2seq_encoder'] = {'_pretrained': {'archive_file': model_archive, 'module_path': '_seq2seq_encoder', 'freeze': False}}
        transfer_model = Model.from_params(vocab=trained_model.vocab, params=Params(model_params))
        for (trained_parameter, transfer_parameter) in zip(trained_model._text_field_embedder.parameters(), transfer_model._text_field_embedder.parameters()):
            assert torch.all(trained_parameter == transfer_parameter)
        for (trained_parameter, transfer_parameter) in zip(trained_model._seq2seq_encoder.parameters(), transfer_model._seq2seq_encoder.parameters()):
            assert torch.all(trained_parameter == transfer_parameter)
        for (trained_parameter, transfer_parameter) in zip(trained_model._feedforward.parameters(), transfer_model._feedforward.parameters()):
            assert torch.all(trained_parameter != transfer_parameter)
        for parameter in transfer_model._text_field_embedder.parameters():
            assert not parameter.requires_grad
        for parameter in transfer_model._seq2seq_encoder.parameters():
            assert parameter.requires_grad

    def test_transferring_of_modules_ensures_type_consistency(self):
        if False:
            while True:
                i = 10
        model_archive = str(self.FIXTURES_ROOT / 'basic_classifier' / 'serialization' / 'model.tar.gz')
        trained_model = load_archive(model_archive).model
        config_file = str(self.FIXTURES_ROOT / 'basic_classifier' / 'experiment_seq2seq.jsonnet')
        model_params = Params.from_file(config_file).pop('model').as_dict(quiet=True)
        model_params['text_field_embedder'] = {'_pretrained': {'archive_file': model_archive, 'module_path': '_seq2seq_encoder._module'}}
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=trained_model.vocab, params=Params(model_params))

    def test_bare_string_params(self):
        if False:
            print('Hello World!')
        reader = DatasetReader.from_params(Params({'type': 'text_classification_json'}))

        class TestLoader(Registrable):

            @classmethod
            def from_partial_objects(cls, data_loader: Lazy[DataLoader]) -> DataLoader:
                if False:
                    print('Hello World!')
                return data_loader.construct(reader=reader, data_path=str(self.FIXTURES_ROOT / 'data' / 'text_classification_json' / 'imdb_corpus2.jsonl'))
        TestLoader.register('test', constructor='from_partial_objects')(TestLoader)
        data_loader = TestLoader.from_params(Params({'type': 'test', 'data_loader': {'batch_size': 2}}))
        assert data_loader.batch_size == 2

    def test_kwargs_are_passed_to_superclass(self):
        if False:
            print('Hello World!')
        params = Params({'type': 'text_classification_json', 'max_instances': 50})
        reader = DatasetReader.from_params(params)
        assert reader.max_instances == 50

    def test_kwargs_with_multiple_inheritance(self):
        if False:
            print('Hello World!')

        class A(Registrable):

            def __init__(self, a: int):
                if False:
                    for i in range(10):
                        print('nop')
                self.a = a
        from numbers import Number

        @A.register('b1')
        class B1(A, Number):

            def __init__(self, b: float, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__(**kwargs)
                self.b = b

        @A.register('b2')
        class B2(Number, A):

            def __init__(self, b: float, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__(**kwargs)
                self.b = b
        b = B1.from_params(params=Params({'a': 4, 'b': 5}))
        assert b.b == 5
        assert b.a == 4
        b = B2.from_params(params=Params({'a': 4, 'b': 5}))
        assert b.b == 5
        assert b.a == 4

    def test_only_infer_superclass_params_if_unknown(self):
        if False:
            while True:
                i = 10
        from allennlp.common.registrable import Registrable

        class BaseClass(Registrable):

            def __init__(self):
                if False:
                    return 10
                self.x = None
                self.a = None
                self.rest = None

        @BaseClass.register('a')
        class A(BaseClass):

            def __init__(self, a: int, x: int, **kwargs):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.x = x
                self.a = a
                self.rest = kwargs

        @BaseClass.register('b')
        class B(A):

            def __init__(self, a: str, x: int=42, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__(x=x, a=-1, raw_a=a, **kwargs)
        params = Params({'type': 'b', 'a': '123'})
        instance = BaseClass.from_params(params)
        assert instance.x == 42
        assert instance.a == -1
        assert len(instance.rest) == 1
        assert type(instance.rest['raw_a']) == str
        assert instance.rest['raw_a'] == '123'

    def test_kwargs_are_passed_to_deeper_superclasses(self):
        if False:
            while True:
                i = 10
        from allennlp.common.registrable import Registrable

        class BaseClass(Registrable):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.a = None
                self.b = None
                self.c = None

        @BaseClass.register('a')
        class A(BaseClass):

            def __init__(self, a: str):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.a = a

        @BaseClass.register('b')
        class B(A):

            def __init__(self, b: str, **kwargs):
                if False:
                    return 10
                super().__init__(**kwargs)
                self.b = b

        @BaseClass.register('c')
        class C(B):

            def __init__(self, c, **kwargs):
                if False:
                    return 10
                super().__init__(**kwargs)
                self.c = c
        params = Params({'type': 'c', 'a': 'a_value', 'b': 'b_value', 'c': 'c_value'})
        instance = BaseClass.from_params(params)
        assert instance.a == 'a_value'
        assert instance.b == 'b_value'
        assert instance.c == 'c_value'

    def test_lazy_construction_can_happen_multiple_times(self):
        if False:
            while True:
                i = 10
        test_string = 'this is a test'
        extra_string = 'extra string'

        class ConstructedObject(FromParams):

            def __init__(self, string: str, extra: str):
                if False:
                    print('Hello World!')
                self.string = string
                self.extra = extra

        class Testing(FromParams):

            def __init__(self, lazy_object: Lazy[ConstructedObject]):
                if False:
                    return 10
                first_time = lazy_object.construct(extra=extra_string)
                second_time = lazy_object.construct(extra=extra_string)
                assert first_time.string == test_string
                assert first_time.extra == extra_string
                assert second_time.string == test_string
                assert second_time.extra == extra_string
        Testing.from_params(Params({'lazy_object': {'string': test_string}}))

    def test_lazy_and_from_params_can_be_pickled(self):
        if False:
            i = 10
            return i + 15
        import pickle
        baz = Baz.from_params(Params({'bar': {'foo': {'a': 2}}}))
        pickle.dumps(baz)

    def test_optional_vs_required_lazy_objects(self):
        if False:
            while True:
                i = 10

        class ConstructedObject(FromParams):

            def __init__(self, a: int):
                if False:
                    while True:
                        i = 10
                self.a = a

        class Testing(FromParams):

            def __init__(self, lazy1: Lazy[ConstructedObject], lazy2: Lazy[ConstructedObject]=Lazy(ConstructedObject), lazy3: Lazy[ConstructedObject]=None, lazy4: Optional[Lazy[ConstructedObject]]=Lazy(ConstructedObject)) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.lazy1 = lazy1.construct()
                self.lazy2 = lazy2.construct(a=2)
                self.lazy3 = None if lazy3 is None else lazy3.construct()
                self.lazy4 = None if lazy4 is None else lazy4.construct(a=1)
        test1 = Testing.from_params(Params({'lazy1': {'a': 1}}))
        assert test1.lazy1.a == 1
        assert test1.lazy2.a == 2
        assert test1.lazy3 is None
        assert test1.lazy4 is not None
        test2 = Testing.from_params(Params({'lazy1': {'a': 1}, 'lazy2': {'a': 3}}))
        assert test2.lazy1.a == 1
        assert test2.lazy2.a == 3
        assert test2.lazy3 is None
        assert test2.lazy4 is not None
        test3 = Testing.from_params(Params({'lazy1': {'a': 1}, 'lazy3': {'a': 3}, 'lazy4': None}))
        assert test3.lazy1.a == 1
        assert test3.lazy2.a == 2
        assert test3.lazy3 is not None
        assert test3.lazy3.a == 3
        assert test3.lazy4 is None
        with pytest.raises(ConfigurationError, match='key "lazy1" is required'):
            Testing.from_params(Params({}))

    def test_iterable(self):
        if False:
            print('Hello World!')
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size: int) -> None:
                if False:
                    i = 10
                    return i + 15
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items: Iterable[A]) -> None:
                if False:
                    while True:
                        i = 10
                self.items = items
        params = Params({'type': 'd', 'items': [{'type': 'b', 'size': 1}, {'type': 'b', 'size': 2}]})
        d = C.from_params(params)
        assert isinstance(d.items, Iterable)
        items = list(d.items)
        assert len(items) == 2
        assert all((isinstance(item, B) for item in items))
        assert items[0].size == 1
        assert items[1].size == 2

    def test_mapping(self):
        if False:
            for i in range(10):
                print('nop')
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register('b')
        class B(A):

            def __init__(self, size: int) -> None:
                if False:
                    while True:
                        i = 10
                self.size = size

        class C(Registrable):
            pass

        @C.register('d')
        class D(C):

            def __init__(self, items: Mapping[str, A]) -> None:
                if False:
                    print('Hello World!')
                self.items = items
        params = Params({'type': 'd', 'items': {'first': {'type': 'b', 'size': 1}, 'second': {'type': 'b', 'size': 2}}})
        d = C.from_params(params)
        assert isinstance(d.items, Mapping)
        assert len(d.items) == 2
        assert all((isinstance(key, str) for key in d.items.keys()))
        assert all((isinstance(value, B) for value in d.items.values()))
        assert d.items['first'].size == 1
        assert d.items['second'].size == 2

    def test_extra_parameters_are_not_allowed_when_there_is_no_constructor(self):
        if False:
            print('Hello World!')

        class A(FromParams):
            pass
        with pytest.raises(ConfigurationError, match='Extra parameters'):
            A.from_params(Params({'some_spurious': 'key', 'value': 'pairs'}))

    def test_explicit_kwargs_always_passed_to_constructor(self):
        if False:
            i = 10
            return i + 15

        class Base(FromParams):

            def __init__(self, lazy: bool=False, x: int=0) -> None:
                if False:
                    return 10
                self.lazy = lazy
                self.x = x

        class A(Base):

            def __init__(self, **kwargs) -> None:
                if False:
                    print('Hello World!')
                assert 'lazy' in kwargs
                super().__init__(**kwargs)
        A.from_params(Params({'lazy': False}))

        class B(Base):

            def __init__(self, **kwargs) -> None:
                if False:
                    while True:
                        i = 10
                super().__init__(lazy=True, **kwargs)
        b = B.from_params(Params({}))
        assert b.lazy is True

    def test_raises_when_there_are_no_implementations(self):
        if False:
            print('Hello World!')

        class A(Registrable):
            pass
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            A.from_params('nonexistent_class')
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            A.from_params(Params({'some_spurious': 'key', 'value': 'pairs'}))
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            A.from_params(Params({}))

        class B(Registrable):

            def __init__(self):
                if False:
                    print('Hello World!')
                pass
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            B.from_params('nonexistent_class')
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            B.from_params(Params({'some_spurious': 'key', 'value': 'pairs'}))
        with pytest.raises(ConfigurationError, match='no registered concrete types'):
            B.from_params(Params({}))

    def test_from_params_raises_error_on_wrong_parameter_name_in_optional_union(self):
        if False:
            while True:
                i = 10

        class NestedClass(FromParams):

            def __init__(self, varname: Optional[str]=None):
                if False:
                    for i in range(10):
                        print('nop')
                self.varname = varname

        class WrapperClass(FromParams):

            def __init__(self, nested_class: Optional[Union[str, NestedClass]]=None):
                if False:
                    print('Hello World!')
                if isinstance(nested_class, str):
                    nested_class = NestedClass(varname=nested_class)
                self.nested_class = nested_class
        with pytest.raises(ConfigurationError):
            WrapperClass.from_params(params=Params({'nested_class': {'wrong_varname': 'varstring'}}))

    def test_from_params_handles_base_class_kwargs(self):
        if False:
            print('Hello World!')

        class Foo(FromParams):

            def __init__(self, a: int, b: str=None, **kwargs) -> None:
                if False:
                    return 10
                self.a = a
                self.b = b
                for (key, value) in kwargs.items():
                    setattr(self, key, value)
        foo = Foo.from_params(Params({'a': 2, 'b': 'hi'}))
        assert foo.a == 2
        assert foo.b == 'hi'
        foo = Foo.from_params(Params({'a': 2, 'b': 'hi', 'c': {'2': '3'}}))
        assert foo.a == 2
        assert foo.b == 'hi'
        assert foo.c == {'2': '3'}

        class Bar(Foo):

            def __init__(self, a: int, b: str, d: int, **kwargs) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__(a, b=b, **kwargs)
                self.d = d
        bar = Bar.from_params(Params({'a': 2, 'b': 'hi', 'c': {'2': '3'}, 'd': 0}))
        assert bar.a == 2
        assert bar.b == 'hi'
        assert bar.c == {'2': '3'}
        assert bar.d == 0

        class Baz(Foo):

            def __init__(self, a: int, b: Optional[str]='a', **kwargs) -> None:
                if False:
                    return 10
                super().__init__(a, b=b, **kwargs)
        baz = Baz.from_params(Params({'a': 2, 'b': None}))
        assert baz.b is None
        baz = Baz.from_params(Params({'a': 2}))
        assert baz.b == 'a'

    def test_from_params_base_class_kwargs_crashes_if_params_not_handled(self):
        if False:
            print('Hello World!')

        class Bar(FromParams):

            def __init__(self, c: str=None) -> None:
                if False:
                    return 10
                self.c = c

        class Foo(Bar):

            def __init__(self, a: int, b: str=None, **kwargs) -> None:
                if False:
                    while True:
                        i = 10
                super().__init__(**kwargs)
                self.a = a
                self.b = b
        foo = Foo.from_params(Params({'a': 2, 'b': 'hi', 'c': 'some value'}))
        assert foo.a == 2
        assert foo.b == 'hi'
        assert foo.c == 'some value'
        with pytest.raises(TypeError, match='invalid_key'):
            Foo.from_params(Params({'a': 2, 'b': 'hi', 'invalid_key': 'some value'}))

    def test_from_params_handles_kwargs_in_non_from_params_registered_class(self):
        if False:
            return 10

        class Bar(Registrable):
            pass

        class Baz:

            def __init__(self, a: int) -> None:
                if False:
                    return 10
                self.a = a

        @Bar.register('foo')
        class Foo(Baz):

            def __init__(self, a: int, b: str=None, **kwargs) -> None:
                if False:
                    while True:
                        i = 10
                super().__init__(a)
                self.b = b
                for (key, value) in kwargs.items():
                    setattr(self, key, value)
        foo = Bar.from_params(Params({'type': 'foo', 'a': 2, 'b': 'hi'}))
        assert foo.a == 2
        assert foo.b == 'hi'
        foo = Bar.from_params(Params({'type': 'foo', 'a': 2, 'b': 'hi', 'c': {'2': '3'}}))
        assert foo.a == 2
        assert foo.b == 'hi'
        assert foo.c == {'2': '3'}

    def test_from_params_does_not_pass_extras_to_non_from_params_registered_class(self):
        if False:
            while True:
                i = 10

        class Bar(Registrable):
            pass

        class Baz:

            def __init__(self, a: int, c: Dict[str, str]=None) -> None:
                if False:
                    return 10
                self.a = a
                self.c = c

        @Bar.register('foo')
        class Foo(Baz):

            def __init__(self, a: int, b: str=None, **kwargs) -> None:
                if False:
                    return 10
                super().__init__(a, **kwargs)
                self.b = b
        foo = Bar.from_params(Params({'type': 'foo', 'a': 2, 'b': 'hi'}))
        assert foo.a == 2
        assert foo.b == 'hi'
        assert foo.c is None
        foo = Bar.from_params(params=Params({'type': 'foo', 'a': 2, 'b': 'hi', 'c': {'2': '3'}}), extra='4')
        assert foo.a == 2
        assert foo.b == 'hi'
        assert foo.c == {'2': '3'}

    def test_from_params_child_has_kwargs_base_implicit_constructor(self):
        if False:
            i = 10
            return i + 15

        class Foo(FromParams):
            pass

        class Bar(Foo):

            def __init__(self, a: int, **kwargs) -> None:
                if False:
                    print('Hello World!')
                self.a = a
        bar = Bar.from_params(Params({'a': 2}))
        assert bar.a == 2

    def test_from_params_has_args(self):
        if False:
            i = 10
            return i + 15

        class Foo(FromParams):

            def __init__(self, a: int, *args) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.a = a
        foo = Foo.from_params(Params({'a': 2}))
        assert foo.a == 2

class MyClass(FromParams):

    def __init__(self, my_int: int, my_bool: bool=False) -> None:
        if False:
            return 10
        self.my_int = my_int
        self.my_bool = my_bool

class Foo(FromParams):

    def __init__(self, a: int=1) -> None:
        if False:
            print('Hello World!')
        self.a = a

class Bar(FromParams):

    def __init__(self, foo: Foo) -> None:
        if False:
            i = 10
            return i + 15
        self.foo = foo

class Baz(FromParams):

    def __init__(self, bar: Lazy[Bar]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._bar = bar

    @property
    def bar(self):
        if False:
            for i in range(10):
                print('nop')
        return self._bar.construct()