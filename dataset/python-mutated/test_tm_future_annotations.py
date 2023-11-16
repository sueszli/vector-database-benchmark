"""This file includes annotation-sensitive tests while having
``from __future__ import annotations`` in effect.

Only tests that don't have an equivalent in ``test_typed_mappings`` are
specified here. All test from ``test_typed_mappings`` are copied over to
the ``test_tm_future_annotations_sync`` by the ``sync_test_file`` script.
"""
from __future__ import annotations
from typing import ClassVar
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
import uuid
from sqlalchemy import exc
from sqlalchemy import ForeignKey
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import select
from sqlalchemy import testing
from sqlalchemy import Uuid
import sqlalchemy.orm
from sqlalchemy.orm import attribute_keyed_dict
from sqlalchemy.orm import KeyFuncDict
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.testing import expect_raises_message
from sqlalchemy.testing import is_
from .test_typed_mapping import expect_annotation_syntax_error
from .test_typed_mapping import MappedColumnTest as _MappedColumnTest
from .test_typed_mapping import RelationshipLHSTest as _RelationshipLHSTest
_R = TypeVar('_R')
M = Mapped

class M3:
    pass

class MappedColumnTest(_MappedColumnTest):

    def test_fully_qualified_mapped_name(self, decl_base):
        if False:
            for i in range(10):
                print('nop')
        'test #8853, regression caused by #8759 ;)\n\n\n        See same test in test_abs_import_only\n\n        '

        class Foo(decl_base):
            __tablename__ = 'foo'
            id: sqlalchemy.orm.Mapped[int] = mapped_column(primary_key=True)
            data: sqlalchemy.orm.Mapped[int] = mapped_column()
            data2: sqlalchemy.orm.Mapped[int]
        self.assert_compile(select(Foo), 'SELECT foo.id, foo.data, foo.data2 FROM foo')

    def test_indirect_mapped_name_module_level(self, decl_base):
        if False:
            return 10
        'test #8759\n\n\n        Note that M by definition has to be at the module level to be\n        valid, and not locally declared here, this is in accordance with\n        mypy::\n\n\n            def make_class() -> None:\n                ll = list\n\n                x: ll[int] = [1, 2, 3]\n\n        Will return::\n\n            $ mypy test3.py\n            test3.py:4: error: Variable "ll" is not valid as a type  [valid-type]\n            test3.py:4: note: See https://mypy.readthedocs.io/en/stable/common_issues.html#variables-vs-type-aliases\n            Found 1 error in 1 file (checked 1 source file)\n\n        Whereas the correct form is::\n\n            ll = list\n\n            def make_class() -> None:\n\n                x: ll[int] = [1, 2, 3]\n\n\n        '

        class Foo(decl_base):
            __tablename__ = 'foo'
            id: M[int] = mapped_column(primary_key=True)
            data: M[int] = mapped_column()
            data2: M[int]
        self.assert_compile(select(Foo), 'SELECT foo.id, foo.data, foo.data2 FROM foo')

    def test_indirect_mapped_name_local_level(self, decl_base):
        if False:
            i = 10
            return i + 15
        'test #8759.\n\n        this should raise an error.\n\n        '
        M2 = Mapped
        with expect_raises_message(exc.ArgumentError, 'Could not interpret annotation M2\\[int\\].  Check that it uses names that are correctly imported at the module level.'):

            class Foo(decl_base):
                __tablename__ = 'foo'
                id: M2[int] = mapped_column(primary_key=True)
                data2: M2[int]

    def test_indirect_mapped_name_itswrong(self, decl_base):
        if False:
            i = 10
            return i + 15
        'test #8759.\n\n        this should raise an error.\n\n        '
        with expect_annotation_syntax_error('Foo.id'):

            class Foo(decl_base):
                __tablename__ = 'foo'
                id: M3[int] = mapped_column(primary_key=True)
                data2: M3[int]

    def test_typ_not_in_cls_namespace(self, decl_base):
        if False:
            return 10
        'test #8742.\n\n        This tests that when types are resolved, they use the ``__module__``\n        of they class they are used within, not the mapped class.\n\n        '

        class Mixin:
            id: Mapped[int] = mapped_column(primary_key=True)
            data: Mapped[uuid.UUID]

        class MyClass(Mixin, decl_base):
            __module__ = 'some.module'
            __tablename__ = 'mytable'
        is_(MyClass.id.expression.type._type_affinity, Integer)
        is_(MyClass.data.expression.type._type_affinity, Uuid)

    def test_dont_ignore_unresolvable(self, decl_base):
        if False:
            for i in range(10):
                print('nop')
        'test #8888'
        with expect_raises_message(exc.ArgumentError, 'Could not resolve all types within mapped annotation: \\"Mapped\\[fake\\]\\".  Ensure all types are written correctly and are imported within the module in use.'):

            class A(decl_base):
                __tablename__ = 'a'
                id: Mapped[int] = mapped_column(primary_key=True)
                data: Mapped[fake]

    @testing.variation('reference_type', ['plain', 'plain_optional', 'container_w_local_mapped', 'container_w_remote_mapped'])
    def test_i_have_a_classvar_on_my_class(self, decl_base, reference_type):
        if False:
            for i in range(10):
                print('nop')
        if reference_type.container_w_remote_mapped:

            class MyOtherClass(decl_base):
                __tablename__ = 'myothertable'
                id: Mapped[int] = mapped_column(primary_key=True)

        class MyClass(decl_base):
            __tablename__ = 'mytable'
            id: Mapped[int] = mapped_column(primary_key=True)
            data: Mapped[str] = mapped_column(default='some default')
            if reference_type.container_w_remote_mapped:
                status: ClassVar[Dict[str, MyOtherClass]]
            elif reference_type.container_w_local_mapped:
                status: ClassVar[Dict[str, MyClass]]
            elif reference_type.plain_optional:
                status: ClassVar[Optional[int]]
            elif reference_type.plain:
                status: ClassVar[int]
        m1 = MyClass(id=1, data=5)
        assert 'status' not in inspect(m1).mapper.attrs

class MappedOneArg(KeyFuncDict[str, _R]):
    pass

class RelationshipLHSTest(_RelationshipLHSTest):

    def test_bidirectional_literal_annotations(self, decl_base):
        if False:
            for i in range(10):
                print('nop')
        'test the \'string cleanup\' function in orm/util.py, where\n        we receive a string annotation like::\n\n            "Mapped[List[B]]"\n\n        Which then fails to evaluate because we don\'t have "B" yet.\n        The annotation is converted on the fly to::\n\n            \'Mapped[List["B"]]\'\n\n        so that when we evaluated it, we get ``Mapped[List["B"]]`` and\n        can extract "B" as a string.\n\n        '

        class A(decl_base):
            __tablename__ = 'a'
            id: Mapped[int] = mapped_column(primary_key=True)
            data: Mapped[str] = mapped_column()
            bs: Mapped[List[B]] = relationship(back_populates='a')

        class B(decl_base):
            __tablename__ = 'b'
            id: Mapped[int] = mapped_column(Integer, primary_key=True)
            a_id: Mapped[int] = mapped_column(ForeignKey('a.id'))
            a: Mapped[A] = relationship(back_populates='bs', primaryjoin=a_id == A.id)
        a1 = A(data='data')
        b1 = B()
        a1.bs.append(b1)
        is_(a1, b1.a)

    def test_collection_class_dict_attr_mapped_collection_literal_annotations(self, decl_base):
        if False:
            print('Hello World!')

        class A(decl_base):
            __tablename__ = 'a'
            id: Mapped[int] = mapped_column(primary_key=True)
            data: Mapped[str] = mapped_column()
            bs: Mapped[KeyFuncDict[str, B]] = relationship(collection_class=attribute_keyed_dict('name'))

        class B(decl_base):
            __tablename__ = 'b'
            id: Mapped[int] = mapped_column(Integer, primary_key=True)
            a_id: Mapped[int] = mapped_column(ForeignKey('a.id'))
            name: Mapped[str] = mapped_column()
        self._assert_dict(A, B)

    def test_collection_cls_attr_mapped_collection_dbl_literal_annotations(self, decl_base):
        if False:
            for i in range(10):
                print('nop')

        class A(decl_base):
            __tablename__ = 'a'
            id: Mapped[int] = mapped_column(primary_key=True)
            data: Mapped[str] = mapped_column()
            bs: Mapped[KeyFuncDict[str, 'B']] = relationship(collection_class=attribute_keyed_dict('name'))

        class B(decl_base):
            __tablename__ = 'b'
            id: Mapped[int] = mapped_column(Integer, primary_key=True)
            a_id: Mapped[int] = mapped_column(ForeignKey('a.id'))
            name: Mapped[str] = mapped_column()
        self._assert_dict(A, B)

    def test_collection_cls_not_locatable(self, decl_base):
        if False:
            print('Hello World!')

        class MyCollection(KeyFuncDict):
            pass
        with expect_raises_message(exc.ArgumentError, "Could not interpret annotation Mapped\\[MyCollection\\['B'\\]\\]."):

            class A(decl_base):
                __tablename__ = 'a'
                id: Mapped[int] = mapped_column(primary_key=True)
                data: Mapped[str] = mapped_column()
                bs: Mapped[MyCollection['B']] = relationship(collection_class=attribute_keyed_dict('name'))

    def test_collection_cls_one_arg(self, decl_base):
        if False:
            while True:
                i = 10

        class A(decl_base):
            __tablename__ = 'a'
            id: Mapped[int] = mapped_column(primary_key=True)
            data: Mapped[str] = mapped_column()
            bs: Mapped[MappedOneArg['B']] = relationship(collection_class=attribute_keyed_dict('name'))

        class B(decl_base):
            __tablename__ = 'b'
            id: Mapped[int] = mapped_column(Integer, primary_key=True)
            a_id: Mapped[int] = mapped_column(ForeignKey('a.id'))
            name: Mapped[str] = mapped_column()
        self._assert_dict(A, B)

    def _assert_dict(self, A, B):
        if False:
            i = 10
            return i + 15
        A.registry.configure()
        a1 = A()
        b1 = B(name='foo')
        a1.bs.set(b1)
        is_(a1.bs['foo'], b1)

    def test_indirect_name_relationship_arg_override(self, decl_base):
        if False:
            i = 10
            return i + 15
        "test #8759\n\n        in this test we assume a case where the type for the Mapped annnotation\n        a. has to be a different name than the actual class name and\n        b. cannot be imported outside of TYPE CHECKING.  user will then put\n        the real name inside of relationship().  we have to succeed even though\n        we can't resolve the annotation.\n\n        "

        class B(decl_base):
            __tablename__ = 'b'
            id: Mapped[int] = mapped_column(Integer, primary_key=True)
            a_id: Mapped[int] = mapped_column(ForeignKey('a.id'))
        if TYPE_CHECKING:
            BNonExistent = B

        class A(decl_base):
            __tablename__ = 'a'
            id: Mapped[int] = mapped_column(primary_key=True)
            data: Mapped[str] = mapped_column()
            bs: Mapped[List[BNonExistent]] = relationship('B')
        self.assert_compile(select(A).join(A.bs), 'SELECT a.id, a.data FROM a JOIN b ON a.id = b.a_id')