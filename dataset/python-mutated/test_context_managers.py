import unittest
import pytest
from mongoengine import *
from mongoengine.connection import get_db
from mongoengine.context_managers import no_dereference, no_sub_classes, query_counter, set_read_write_concern, set_write_concern, switch_collection, switch_db
from mongoengine.pymongo_support import count_documents

class TestContextManagers:

    def test_set_write_concern(self):
        if False:
            for i in range(10):
                print('nop')
        connect('mongoenginetest')

        class User(Document):
            name = StringField()
        collection = User._get_collection()
        original_write_concern = collection.write_concern
        with set_write_concern(collection, {'w': 'majority', 'j': True, 'wtimeout': 1234}) as updated_collection:
            assert updated_collection.write_concern.document == {'w': 'majority', 'j': True, 'wtimeout': 1234}
        assert original_write_concern.document == collection.write_concern.document

    def test_set_read_write_concern(self):
        if False:
            while True:
                i = 10
        connect('mongoenginetest')

        class User(Document):
            name = StringField()
        collection = User._get_collection()
        original_read_concern = collection.read_concern
        original_write_concern = collection.write_concern
        with set_read_write_concern(collection, {'w': 'majority', 'j': True, 'wtimeout': 1234}, {'level': 'local'}) as update_collection:
            assert update_collection.read_concern.document == {'level': 'local'}
            assert update_collection.write_concern.document == {'w': 'majority', 'j': True, 'wtimeout': 1234}
        assert original_read_concern.document == collection.read_concern.document
        assert original_write_concern.document == collection.write_concern.document

    def test_switch_db_context_manager(self):
        if False:
            i = 10
            return i + 15
        connect('mongoenginetest')
        register_connection('testdb-1', 'mongoenginetest2')

        class Group(Document):
            name = StringField()
        Group.drop_collection()
        Group(name='hello - default').save()
        assert 1 == Group.objects.count()
        with switch_db(Group, 'testdb-1') as Group:
            assert 0 == Group.objects.count()
            Group(name='hello').save()
            assert 1 == Group.objects.count()
            Group.drop_collection()
            assert 0 == Group.objects.count()
        assert 1 == Group.objects.count()

    def test_switch_collection_context_manager(self):
        if False:
            for i in range(10):
                print('nop')
        connect('mongoenginetest')
        register_connection(alias='testdb-1', db='mongoenginetest2')

        class Group(Document):
            name = StringField()
        Group.drop_collection()
        with switch_collection(Group, 'group1') as Group:
            Group.drop_collection()
        Group(name='hello - group').save()
        assert 1 == Group.objects.count()
        with switch_collection(Group, 'group1') as Group:
            assert 0 == Group.objects.count()
            Group(name='hello - group1').save()
            assert 1 == Group.objects.count()
            Group.drop_collection()
            assert 0 == Group.objects.count()
        assert 1 == Group.objects.count()

    def test_no_dereference_context_manager_object_id(self):
        if False:
            i = 10
            return i + 15
        "Ensure that DBRef items in ListFields aren't dereferenced."
        connect('mongoenginetest')

        class User(Document):
            name = StringField()

        class Group(Document):
            ref = ReferenceField(User, dbref=False)
            generic = GenericReferenceField()
            members = ListField(ReferenceField(User, dbref=False))
        User.drop_collection()
        Group.drop_collection()
        for i in range(1, 51):
            User(name='user %s' % i).save()
        user = User.objects.first()
        Group(ref=user, members=User.objects, generic=user).save()
        with no_dereference(Group) as NoDeRefGroup:
            assert Group._fields['members']._auto_dereference
            assert not NoDeRefGroup._fields['members']._auto_dereference
        with no_dereference(Group) as Group:
            group = Group.objects.first()
            for m in group.members:
                assert not isinstance(m, User)
            assert not isinstance(group.ref, User)
            assert not isinstance(group.generic, User)
        for m in group.members:
            assert isinstance(m, User)
        assert isinstance(group.ref, User)
        assert isinstance(group.generic, User)

    def test_no_dereference_context_manager_dbref(self):
        if False:
            i = 10
            return i + 15
        "Ensure that DBRef items in ListFields aren't dereferenced."
        connect('mongoenginetest')

        class User(Document):
            name = StringField()

        class Group(Document):
            ref = ReferenceField(User, dbref=True)
            generic = GenericReferenceField()
            members = ListField(ReferenceField(User, dbref=True))
        User.drop_collection()
        Group.drop_collection()
        for i in range(1, 51):
            User(name='user %s' % i).save()
        user = User.objects.first()
        Group(ref=user, members=User.objects, generic=user).save()
        with no_dereference(Group) as NoDeRefGroup:
            assert Group._fields['members']._auto_dereference
            assert not NoDeRefGroup._fields['members']._auto_dereference
        with no_dereference(Group) as Group:
            group = Group.objects.first()
            assert all((not isinstance(m, User) for m in group.members))
            assert not isinstance(group.ref, User)
            assert not isinstance(group.generic, User)
        assert all((isinstance(m, User) for m in group.members))
        assert isinstance(group.ref, User)
        assert isinstance(group.generic, User)

    def test_no_sub_classes(self):
        if False:
            i = 10
            return i + 15

        class A(Document):
            x = IntField()
            meta = {'allow_inheritance': True}

        class B(A):
            z = IntField()

        class C(B):
            zz = IntField()
        A.drop_collection()
        A(x=10).save()
        A(x=15).save()
        B(x=20).save()
        B(x=30).save()
        C(x=40).save()
        assert A.objects.count() == 5
        assert B.objects.count() == 3
        assert C.objects.count() == 1
        with no_sub_classes(A):
            assert A.objects.count() == 2
            for obj in A.objects:
                assert obj.__class__ == A
        with no_sub_classes(B):
            assert B.objects.count() == 2
            for obj in B.objects:
                assert obj.__class__ == B
        with no_sub_classes(C):
            assert C.objects.count() == 1
            for obj in C.objects:
                assert obj.__class__ == C
        assert A.objects.count() == 5
        assert B.objects.count() == 3
        assert C.objects.count() == 1

    def test_no_sub_classes_modification_to_document_class_are_temporary(self):
        if False:
            for i in range(10):
                print('nop')

        class A(Document):
            x = IntField()
            meta = {'allow_inheritance': True}

        class B(A):
            z = IntField()
        assert A._subclasses == ('A', 'A.B')
        with no_sub_classes(A):
            assert A._subclasses == ('A',)
        assert A._subclasses == ('A', 'A.B')
        assert B._subclasses == ('A.B',)
        with no_sub_classes(B):
            assert B._subclasses == ('A.B',)
        assert B._subclasses == ('A.B',)

    def test_no_subclass_context_manager_does_not_swallow_exception(self):
        if False:
            i = 10
            return i + 15

        class User(Document):
            name = StringField()
        with pytest.raises(TypeError):
            with no_sub_classes(User):
                raise TypeError()

    def test_query_counter_does_not_swallow_exception(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError):
            with query_counter():
                raise TypeError()

    def test_query_counter_temporarily_modifies_profiling_level(self):
        if False:
            for i in range(10):
                print('nop')
        connect('mongoenginetest')
        db = get_db()

        def _current_profiling_level():
            if False:
                while True:
                    i = 10
            return db.command({'profile': -1})['was']

        def _set_profiling_level(lvl):
            if False:
                i = 10
                return i + 15
            db.command({'profile': lvl})
        initial_profiling_level = _current_profiling_level()
        try:
            new_level = 1
            _set_profiling_level(new_level)
            assert _current_profiling_level() == new_level
            with query_counter():
                assert _current_profiling_level() == 2
            assert _current_profiling_level() == new_level
        except Exception:
            _set_profiling_level(initial_profiling_level)
            raise

    def test_query_counter(self):
        if False:
            i = 10
            return i + 15
        connect('mongoenginetest')
        db = get_db()
        collection = db.query_counter
        collection.drop()

        def issue_1_count_query():
            if False:
                return 10
            count_documents(collection, {})

        def issue_1_insert_query():
            if False:
                i = 10
                return i + 15
            collection.insert_one({'test': 'garbage'})

        def issue_1_find_query():
            if False:
                return 10
            collection.find_one()
        counter = 0
        with query_counter() as q:
            assert q == counter
            assert q == counter
            for _ in range(10):
                issue_1_insert_query()
                counter += 1
            assert q == counter
            for _ in range(4):
                issue_1_find_query()
                counter += 1
            assert q == counter
            for _ in range(3):
                issue_1_count_query()
                counter += 1
            assert q == counter
            assert int(q) == counter
            assert repr(q) == str(int(q))
            assert q > -1
            assert q >= int(q)
            assert q != -1
            assert q < 1000
            assert q <= int(q)

    def test_query_counter_alias(self):
        if False:
            return 10
        'query_counter works properly with db aliases?'
        register_connection('testdb-1', 'mongoenginetest2')

        class A(Document):
            """Uses default db_alias"""
            name = StringField()

        class B(Document):
            """Uses testdb-1 db_alias"""
            name = StringField()
            meta = {'db_alias': 'testdb-1'}
        A.drop_collection()
        B.drop_collection()
        with query_counter() as q:
            assert q == 0
            A.objects.create(name='A')
            assert q == 1
            a = A.objects.first()
            assert q == 2
            a.name = 'Test A'
            a.save()
            assert q == 3
            B.objects().first()
            assert q == 3
        with query_counter(alias='testdb-1') as q:
            assert q == 0
            B.objects.create(name='B')
            assert q == 1
            b = B.objects.first()
            assert q == 2
            b.name = 'Test B'
            b.save()
            assert b.name == 'Test B'
            assert q == 3
            A.objects().first()
            assert q == 3

    def test_query_counter_counts_getmore_queries(self):
        if False:
            return 10
        connect('mongoenginetest')
        db = get_db()
        collection = db.query_counter
        collection.drop()
        many_docs = [{'test': 'garbage %s' % i} for i in range(150)]
        collection.insert_many(many_docs)
        with query_counter() as q:
            assert q == 0
            list(collection.find())
            assert q == 2

    def test_query_counter_ignores_particular_queries(self):
        if False:
            for i in range(10):
                print('nop')
        connect('mongoenginetest')
        db = get_db()
        collection = db.query_counter
        collection.insert_many([{'test': 'garbage %s' % i} for i in range(10)])
        with query_counter() as q:
            assert q == 0
            cursor = collection.find()
            assert q == 0
            _ = next(cursor)
            assert q == 1
            cursor.close()
            assert q == 1
            _ = db.system.indexes.find_one()
            assert q == 1
if __name__ == '__main__':
    unittest.main()