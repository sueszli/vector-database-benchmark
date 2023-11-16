import datetime
import unittest
import uuid
from decimal import Decimal
import pymongo
import pytest
from bson import DBRef, ObjectId
from pymongo.read_preferences import ReadPreference
from pymongo.results import UpdateResult
from mongoengine import *
from mongoengine.connection import get_db
from mongoengine.context_managers import query_counter, switch_db
from mongoengine.errors import InvalidQueryError
from mongoengine.mongodb_support import MONGODB_36, get_mongodb_version
from mongoengine.queryset import DoesNotExist, MultipleObjectsReturned, QuerySet, QuerySetManager, queryset_manager
from mongoengine.queryset.base import BaseQuerySet
from tests.utils import requires_mongodb_gte_42, requires_mongodb_gte_44, requires_mongodb_lt_42

class db_ops_tracker(query_counter):

    def get_ops(self):
        if False:
            return 10
        ignore_query = dict(self._ignored_query)
        ignore_query['command.count'] = {'$ne': 'system.profile'}
        return list(self.db.system.profile.find(ignore_query))

def get_key_compat(mongo_ver):
    if False:
        return 10
    ORDER_BY_KEY = 'sort'
    CMD_QUERY_KEY = 'command' if mongo_ver >= MONGODB_36 else 'query'
    return (ORDER_BY_KEY, CMD_QUERY_KEY)

class TestQueryset(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        connect(db='mongoenginetest')
        connect(db='mongoenginetest2', alias='test2')

        class PersonMeta(EmbeddedDocument):
            weight = IntField()

        class Person(Document):
            name = StringField()
            age = IntField()
            person_meta = EmbeddedDocumentField(PersonMeta)
            meta = {'allow_inheritance': True}
        Person.drop_collection()
        self.PersonMeta = PersonMeta
        self.Person = Person
        self.mongodb_version = get_mongodb_version()

    def test_initialisation(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that a QuerySet is correctly initialised by QuerySetManager.'
        assert isinstance(self.Person.objects, QuerySet)
        assert self.Person.objects._collection.name == self.Person._get_collection_name()
        assert isinstance(self.Person.objects._collection, pymongo.collection.Collection)

    def test_cannot_perform_joins_references(self):
        if False:
            while True:
                i = 10

        class BlogPost(Document):
            author = ReferenceField(self.Person)
            author2 = GenericReferenceField()
        with pytest.raises(InvalidQueryError):
            list(BlogPost.objects(author__name='test'))
        with pytest.raises(InvalidQueryError):
            list(BlogPost.objects(author2__name='test'))

    def test_find(self):
        if False:
            print('Hello World!')
        'Ensure that a query returns a valid set of results.'
        user_a = self.Person.objects.create(name='User A', age=20)
        user_b = self.Person.objects.create(name='User B', age=30)
        people = self.Person.objects
        assert people.count() == 2
        results = list(people)
        assert isinstance(results[0], self.Person)
        assert isinstance(results[0].id, ObjectId)
        assert results[0] == user_a
        assert results[0].name == 'User A'
        assert results[0].age == 20
        assert results[1] == user_b
        assert results[1].name == 'User B'
        assert results[1].age == 30
        people = self.Person.objects(age=20)
        assert people.count() == 1
        person = next(people)
        assert person == user_a
        assert person.name == 'User A'
        assert person.age == 20

    def test_slicing_sets_empty_limit_skip(self):
        if False:
            i = 10
            return i + 15
        self.Person.objects.insert([self.Person(name=f'User {i}', age=i) for i in range(5)], load_bulk=False)
        self.Person.objects.create(name='User B', age=30)
        self.Person.objects.create(name='User C', age=40)
        qs = self.Person.objects()[1:2]
        assert (qs._empty, qs._skip, qs._limit) == (False, 1, 1)
        assert len(list(qs)) == 1
        qs = self.Person.objects()[1:1]
        assert (qs._empty, qs._skip, qs._limit) == (True, 1, 0)
        assert len(list(qs)) == 0
        qs2 = qs[1:5]
        assert (qs2._empty, qs2._skip, qs2._limit) == (False, 1, 4)
        assert len(list(qs2)) == 4

    def test_limit_0_returns_all_documents(self):
        if False:
            return 10
        self.Person.objects.create(name='User A', age=20)
        self.Person.objects.create(name='User B', age=30)
        n_docs = self.Person.objects().count()
        persons = list(self.Person.objects().limit(0))
        assert len(persons) == 2 == n_docs

    def test_limit_0(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that QuerySet.limit works as expected.'
        self.Person.objects.create(name='User A', age=20)
        qs = self.Person.objects.limit(0)
        assert qs.count() == 0

    def test_limit(self):
        if False:
            i = 10
            return i + 15
        'Ensure that QuerySet.limit works as expected.'
        user_a = self.Person.objects.create(name='User A', age=20)
        _ = self.Person.objects.create(name='User B', age=30)
        people = list(self.Person.objects.limit(1))
        assert len(people) == 1
        assert people[0] == user_a
        people = self.Person.objects
        assert len(people) == 2
        people2 = people.limit(1)
        assert len(people) == 2
        assert len(people2) == 1
        assert people2[0] == user_a
        people = self.Person.objects.limit(0)
        assert people.count(with_limit_and_skip=True) == 2
        assert len(people) == 2
        person = self.Person.objects().limit(1).only('name').first()
        assert person == user_a
        assert person.name == 'User A'
        assert person.age is None

    def test_skip(self):
        if False:
            return 10
        'Ensure that QuerySet.skip works as expected.'
        user_a = self.Person.objects.create(name='User A', age=20)
        user_b = self.Person.objects.create(name='User B', age=30)
        people = list(self.Person.objects.skip(0))
        assert len(people) == 2
        assert people[0] == user_a
        assert people[1] == user_b
        people = list(self.Person.objects.skip(1))
        assert len(people) == 1
        assert people[0] == user_b
        people = self.Person.objects
        assert len(people) == 2
        people2 = people.skip(1)
        assert len(people) == 2
        assert len(people2) == 1
        assert people2[0] == user_b
        person = self.Person.objects().skip(1).only('name').first()
        assert person == user_b
        assert person.name == 'User B'
        assert person.age is None

    def test___getitem___invalid_index(self):
        if False:
            i = 10
            return i + 15
        'Ensure slicing a queryset works as expected.'
        with pytest.raises(TypeError):
            self.Person.objects()['a']

    def test_slice(self):
        if False:
            return 10
        'Ensure slicing a queryset works as expected.'
        user_a = self.Person.objects.create(name='User A', age=20)
        user_b = self.Person.objects.create(name='User B', age=30)
        user_c = self.Person.objects.create(name='User C', age=40)
        people = list(self.Person.objects[:2])
        assert len(people) == 2
        assert people[0] == user_a
        assert people[1] == user_b
        people = list(self.Person.objects[1:])
        assert len(people) == 2
        assert people[0] == user_b
        assert people[1] == user_c
        people = list(self.Person.objects[1:2])
        assert len(people) == 1
        assert people[0] == user_b
        people = self.Person.objects
        assert len(people) == 3
        people2 = people[1:2]
        assert len(people2) == 1
        assert people2[0] == user_b
        qs = self.Person.objects[1:2]
        qs._cursor
        qs._cursor_obj = None
        people = list(qs)
        assert len(people) == 1
        assert people[0].name == 'User B'
        people = list(self.Person.objects[1:1])
        assert len(people) == 0
        people = list(self.Person.objects[80000:80001])
        assert len(people) == 0
        self.Person.objects.delete()
        for i in range(55):
            self.Person(name='A%s' % i, age=i).save()
        assert self.Person.objects.count() == 55
        assert 'Person object' == '%s' % self.Person.objects[0]
        assert '[<Person: Person object>, <Person: Person object>]' == '%s' % self.Person.objects[1:3]
        assert '[<Person: Person object>, <Person: Person object>]' == '%s' % self.Person.objects[51:53]

    def test_find_one(self):
        if False:
            while True:
                i = 10
        'Ensure that a query using find_one returns a valid result.'
        person1 = self.Person(name='User A', age=20)
        person1.save()
        person2 = self.Person(name='User B', age=30)
        person2.save()
        person = self.Person.objects.first()
        assert isinstance(person, self.Person)
        assert person.name == 'User A'
        assert person.age == 20
        person = self.Person.objects(age=30).first()
        assert person.name == 'User B'
        person = self.Person.objects(age__lt=30).first()
        assert person.name == 'User A'
        person = self.Person.objects[0]
        assert person.name == 'User A'
        person = self.Person.objects[1]
        assert person.name == 'User B'
        with pytest.raises(IndexError):
            self.Person.objects[2]
        person = self.Person.objects.with_id(person1.id)
        assert person.name == 'User A'
        with pytest.raises(InvalidQueryError):
            self.Person.objects(name='User A').with_id(person1.id)

    def test_get_no_document_exists_raises_doesnotexist(self):
        if False:
            while True:
                i = 10
        assert self.Person.objects.count() == 0
        with pytest.raises(DoesNotExist):
            self.Person.objects.get()
        with pytest.raises(self.Person.DoesNotExist):
            self.Person.objects.get()

    def test_get_multiple_match_raises_multipleobjectsreturned(self):
        if False:
            i = 10
            return i + 15
        'Ensure that a query using ``get`` returns at most one result.'
        assert self.Person.objects().count() == 0
        person1 = self.Person(name='User A', age=20)
        person1.save()
        p = self.Person.objects.get()
        assert p == person1
        person2 = self.Person(name='User B', age=20)
        person2.save()
        person3 = self.Person(name='User C', age=30)
        person3.save()
        with pytest.raises(MultipleObjectsReturned):
            self.Person.objects.get()
        with pytest.raises(self.Person.MultipleObjectsReturned):
            self.Person.objects.get()
        with pytest.raises(MultipleObjectsReturned):
            self.Person.objects.get(age__lt=30)
        with pytest.raises(MultipleObjectsReturned) as exc_info:
            self.Person.objects(age__lt=30).get()
        assert '2 or more items returned, instead of 1' == str(exc_info.value)
        person = self.Person.objects.get(age=30)
        assert person == person3

    def test_find_array_position(self):
        if False:
            print('Hello World!')
        'Ensure that query by array position works.'

        class Comment(EmbeddedDocument):
            name = StringField()

        class Post(EmbeddedDocument):
            comments = ListField(EmbeddedDocumentField(Comment))

        class Blog(Document):
            tags = ListField(StringField())
            posts = ListField(EmbeddedDocumentField(Post))
        Blog.drop_collection()
        Blog.objects.create(tags=['a', 'b'])
        assert Blog.objects(tags__0='a').count() == 1
        assert Blog.objects(tags__0='b').count() == 0
        assert Blog.objects(tags__1='a').count() == 0
        assert Blog.objects(tags__1='b').count() == 1
        Blog.drop_collection()
        comment1 = Comment(name='testa')
        comment2 = Comment(name='testb')
        post1 = Post(comments=[comment1, comment2])
        post2 = Post(comments=[comment2, comment2])
        blog1 = Blog.objects.create(posts=[post1, post2])
        blog2 = Blog.objects.create(posts=[post2, post1])
        blog = Blog.objects(posts__0__comments__0__name='testa').get()
        assert blog == blog1
        blog = Blog.objects(posts__0__comments__0__name='testb').get()
        assert blog == blog2
        query = Blog.objects(posts__1__comments__1__name='testb')
        assert query.count() == 2
        query = Blog.objects(posts__1__comments__1__name='testa')
        assert query.count() == 0
        query = Blog.objects(posts__0__comments__1__name='testa')
        assert query.count() == 0
        Blog.drop_collection()

    def test_none(self):
        if False:
            i = 10
            return i + 15

        class A(Document):
            s = StringField()
        A.drop_collection()
        A().save()
        assert A.objects.count() == 1
        assert A.objects.none().update(s='1') == 0
        assert A.objects.none().update_one(s='1') == 0
        assert A.objects.none().modify(s='1') is None
        assert A.objects(s='1').count() == 0
        assert A.objects.none().first() is None
        assert list(A.objects.none()) == []
        assert list(A.objects.none().all()) == []
        assert list(A.objects.none().limit(1)) == []
        assert list(A.objects.none().skip(1)) == []
        assert list(A.objects.none()[:5]) == []

    def test_chaining(self):
        if False:
            while True:
                i = 10

        class A(Document):
            s = StringField()

        class B(Document):
            ref = ReferenceField(A)
            boolfield = BooleanField(default=False)
        A.drop_collection()
        B.drop_collection()
        a1 = A(s='test1').save()
        a2 = A(s='test2').save()
        B(ref=a1, boolfield=True).save()
        q1 = B.objects.filter(ref__in=[a1, a2], ref=a1)._query
        q2 = B.objects.filter(ref__in=[a1, a2])
        q2 = q2.filter(ref=a1)._query
        assert q1 == q2
        a_objects = A.objects(s='test1')
        query = B.objects(ref__in=a_objects)
        query = query.filter(boolfield=True)
        assert query.count() == 1

    def test_batch_size(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that batch_size works.'

        class A(Document):
            s = StringField()
        A.drop_collection()
        for i in range(100):
            A.objects.create(s=str(i))
        cnt = 0
        for _ in A.objects.batch_size(10):
            cnt += 1
        assert cnt == 100
        qs = A.objects.all()
        qs = qs.limit(10).batch_size(20).skip(91)
        cnt = 0
        for _ in qs:
            cnt += 1
        assert cnt == 9
        qs = A.objects.batch_size(-1)
        with pytest.raises(ValueError):
            list(qs)

    def test_batch_size_cloned(self):
        if False:
            while True:
                i = 10

        class A(Document):
            s = StringField()
        qs = A.objects.batch_size(5)
        assert qs._batch_size == 5
        qs_clone = qs.clone()
        assert qs_clone._batch_size == 5

    def test_update_write_concern(self):
        if False:
            i = 10
            return i + 15
        'Test that passing write_concern works'
        self.Person.drop_collection()
        write_concern = {'fsync': True}
        author = self.Person.objects.create(name='Test User')
        author.save(write_concern=write_concern)
        author = self.Person(name='Test User2')
        author.save(write_concern=None)
        result = self.Person.objects.update(set__name='Ross', write_concern={'w': 1})
        assert result == 2
        result = self.Person.objects.update(set__name='Ross', write_concern={'w': 0})
        assert result is None
        result = self.Person.objects.update_one(set__name='Test User', write_concern={'w': 1})
        assert result == 1
        result = self.Person.objects.update_one(set__name='Test User', write_concern={'w': 0})
        assert result is None

    def test_update_update_has_a_value(self):
        if False:
            for i in range(10):
                print('nop')
        'Test to ensure that update is passed a value to update to'
        self.Person.drop_collection()
        author = self.Person.objects.create(name='Test User')
        with pytest.raises(OperationError):
            self.Person.objects(pk=author.pk).update({})
        with pytest.raises(OperationError):
            self.Person.objects(pk=author.pk).update_one({})

    def test_update_array_position(self):
        if False:
            i = 10
            return i + 15
        'Ensure that updating by array position works.\n\n        Check update() and update_one() can take syntax like:\n            set__posts__1__comments__1__name="testc"\n        Check that it only works for ListFields.\n        '

        class Comment(EmbeddedDocument):
            name = StringField()

        class Post(EmbeddedDocument):
            comments = ListField(EmbeddedDocumentField(Comment))

        class Blog(Document):
            tags = ListField(StringField())
            posts = ListField(EmbeddedDocumentField(Post))
        Blog.drop_collection()
        comment1 = Comment(name='testa')
        comment2 = Comment(name='testb')
        post1 = Post(comments=[comment1, comment2])
        post2 = Post(comments=[comment2, comment2])
        Blog.objects.create(posts=[post1, post2])
        Blog.objects.create(posts=[post2, post1])
        Blog.objects().update(set__posts__1__comments__0__name='testc')
        testc_blogs = Blog.objects(posts__1__comments__0__name='testc')
        assert testc_blogs.count() == 2
        Blog.drop_collection()
        Blog.objects.create(posts=[post1, post2])
        Blog.objects.create(posts=[post2, post1])
        Blog.objects().update_one(set__posts__1__comments__1__name='testc')
        testc_blogs = Blog.objects(posts__1__comments__1__name='testc')
        assert testc_blogs.count() == 1
        with pytest.raises(InvalidQueryError):
            Blog.objects().update(set__posts__1__comments__0__name__1='asdf')
        Blog.drop_collection()

    def test_update_using_positional_operator(self):
        if False:
            while True:
                i = 10
        'Ensure that the list fields can be updated using the positional\n        operator.'

        class Comment(EmbeddedDocument):
            by = StringField()
            votes = IntField()

        class BlogPost(Document):
            title = StringField()
            comments = ListField(EmbeddedDocumentField(Comment))
        BlogPost.drop_collection()
        c1 = Comment(by='joe', votes=3)
        c2 = Comment(by='jane', votes=7)
        BlogPost(title='ABC', comments=[c1, c2]).save()
        BlogPost.objects(comments__by='jane').update(inc__comments__S__votes=1)
        post = BlogPost.objects.first()
        assert post.comments[1].by == 'jane'
        assert post.comments[1].votes == 8

    def test_update_using_positional_operator_matches_first(self):
        if False:
            i = 10
            return i + 15

        class Simple(Document):
            x = ListField()
        Simple.drop_collection()
        Simple(x=[1, 2, 3, 2]).save()
        Simple.objects(x=2).update(inc__x__S=1)
        simple = Simple.objects.first()
        assert simple.x == [1, 3, 3, 2]
        Simple.drop_collection()
        Simple.drop_collection()
        Simple(x=[1, 2, 3, 4]).save()
        Simple(x=[2, 3, 4, 5]).save()
        Simple(x=[3, 4, 5, 6]).save()
        Simple(x=[4, 5, 6, 7]).save()
        Simple.objects(x=3).update(set__x__S=0)
        s = Simple.objects()
        assert s[0].x == [1, 2, 0, 4]
        assert s[1].x == [2, 0, 4, 5]
        assert s[2].x == [0, 4, 5, 6]
        assert s[3].x == [4, 5, 6, 7]
        Simple.drop_collection()
        Simple(x=[1, 2, 3, 4, 3, 2, 3, 4]).save()
        Simple.objects(x=3).update(unset__x__S=1)
        simple = Simple.objects.first()
        assert simple.x == [1, 2, None, 4, 3, 2, 3, 4]
        with pytest.raises(OperationError):
            Simple.drop_collection()
            Simple(x=[{'test': [1, 2, 3, 4]}]).save()
            Simple.objects(x__test=2).update(set__x__S__test__S=3)
            assert simple.x == [1, 2, 3, 4]

    def test_update_using_positional_operator_embedded_document(self):
        if False:
            print('Hello World!')
        'Ensure that the embedded documents can be updated using the positional\n        operator.'

        class Vote(EmbeddedDocument):
            score = IntField()

        class Comment(EmbeddedDocument):
            by = StringField()
            votes = EmbeddedDocumentField(Vote)

        class BlogPost(Document):
            title = StringField()
            comments = ListField(EmbeddedDocumentField(Comment))
        BlogPost.drop_collection()
        c1 = Comment(by='joe', votes=Vote(score=3))
        c2 = Comment(by='jane', votes=Vote(score=7))
        BlogPost(title='ABC', comments=[c1, c2]).save()
        BlogPost.objects(comments__by='joe').update(set__comments__S__votes=Vote(score=4))
        post = BlogPost.objects.first()
        assert post.comments[0].by == 'joe'
        assert post.comments[0].votes.score == 4

    def test_update_min_max(self):
        if False:
            i = 10
            return i + 15

        class Scores(Document):
            high_score = IntField()
            low_score = IntField()
        scores = Scores.objects.create(high_score=800, low_score=200)
        Scores.objects(id=scores.id).update(min__low_score=150)
        assert Scores.objects.get(id=scores.id).low_score == 150
        Scores.objects(id=scores.id).update(min__low_score=250)
        assert Scores.objects.get(id=scores.id).low_score == 150
        Scores.objects(id=scores.id).update(max__high_score=1000)
        assert Scores.objects.get(id=scores.id).high_score == 1000
        Scores.objects(id=scores.id).update(max__high_score=500)
        assert Scores.objects.get(id=scores.id).high_score == 1000

    def test_update_multiple(self):
        if False:
            i = 10
            return i + 15

        class Product(Document):
            item = StringField()
            price = FloatField()
        product = Product.objects.create(item='ABC', price=10.99)
        product = Product.objects.create(item='ABC', price=10.99)
        Product.objects(id=product.id).update(mul__price=1.25)
        assert Product.objects.get(id=product.id).price == 13.7375
        unknown_product = Product.objects.create(item='Unknown')
        Product.objects(id=unknown_product.id).update(mul__price=100)
        assert Product.objects.get(id=unknown_product.id).price == 0

    def test_updates_can_have_match_operators(self):
        if False:
            for i in range(10):
                print('nop')

        class Comment(EmbeddedDocument):
            content = StringField()
            name = StringField(max_length=120)
            vote = IntField()

        class Post(Document):
            title = StringField(required=True)
            tags = ListField(StringField())
            comments = ListField(EmbeddedDocumentField('Comment'))
        Post.drop_collection()
        comm1 = Comment(content='very funny indeed', name='John S', vote=1)
        comm2 = Comment(content='kind of funny', name='Mark P', vote=0)
        Post(title='Fun with MongoEngine', tags=['mongodb', 'mongoengine'], comments=[comm1, comm2]).save()
        Post.objects().update_one(pull__comments__vote__lt=1)
        assert 1 == len(Post.objects.first().comments)

    def test_mapfield_update(self):
        if False:
            return 10
        'Ensure that the MapField can be updated.'

        class Member(EmbeddedDocument):
            gender = StringField()
            age = IntField()

        class Club(Document):
            members = MapField(EmbeddedDocumentField(Member))
        Club.drop_collection()
        club = Club()
        club.members['John'] = Member(gender='M', age=13)
        club.save()
        Club.objects().update(set__members={'John': Member(gender='F', age=14)})
        club = Club.objects().first()
        assert club.members['John'].gender == 'F'
        assert club.members['John'].age == 14

    def test_dictfield_update(self):
        if False:
            i = 10
            return i + 15
        'Ensure that the DictField can be updated.'

        class Club(Document):
            members = DictField()
        club = Club()
        club.members['John'] = {'gender': 'M', 'age': 13}
        club.save()
        Club.objects().update(set__members={'John': {'gender': 'F', 'age': 14}})
        club = Club.objects().first()
        assert club.members['John']['gender'] == 'F'
        assert club.members['John']['age'] == 14

    def test_update_results(self):
        if False:
            print('Hello World!')
        self.Person.drop_collection()
        result = self.Person(name='Bob', age=25).update(upsert=True, full_result=True)
        assert isinstance(result, UpdateResult)
        assert 'upserted' in result.raw_result
        assert not result.raw_result['updatedExisting']
        bob = self.Person.objects.first()
        result = bob.update(set__age=30, full_result=True)
        assert isinstance(result, UpdateResult)
        assert result.raw_result['updatedExisting']
        self.Person(name='Bob', age=20).save()
        result = self.Person.objects(name='Bob').update(set__name='bobby', multi=True)
        assert result == 2

    def test_update_validate(self):
        if False:
            for i in range(10):
                print('nop')

        class EmDoc(EmbeddedDocument):
            str_f = StringField()

        class Doc(Document):
            str_f = StringField()
            dt_f = DateTimeField()
            cdt_f = ComplexDateTimeField()
            ed_f = EmbeddedDocumentField(EmDoc)
        with pytest.raises(ValidationError):
            Doc.objects().update(str_f=1, upsert=True)
        with pytest.raises(ValidationError):
            Doc.objects().update(dt_f='datetime', upsert=True)
        with pytest.raises(ValidationError):
            Doc.objects().update(ed_f__str_f=1, upsert=True)

    def test_update_related_models(self):
        if False:
            return 10

        class TestPerson(Document):
            name = StringField()

        class TestOrganization(Document):
            name = StringField()
            owner = ReferenceField(TestPerson)
        TestPerson.drop_collection()
        TestOrganization.drop_collection()
        p = TestPerson(name='p1')
        p.save()
        o = TestOrganization(name='o1')
        o.save()
        o.owner = p
        p.name = 'p2'
        assert o._get_changed_fields() == ['owner']
        assert p._get_changed_fields() == ['name']
        o.save()
        assert o._get_changed_fields() == []
        assert p._get_changed_fields() == ['name']
        p.save()
        p.reload()
        assert p.name == 'p2'

    def test_upsert(self):
        if False:
            while True:
                i = 10
        self.Person.drop_collection()
        self.Person.objects(pk=ObjectId(), name='Bob', age=30).update(upsert=True)
        bob = self.Person.objects.first()
        assert 'Bob' == bob.name
        assert 30 == bob.age

    def test_upsert_one(self):
        if False:
            return 10
        self.Person.drop_collection()
        bob = self.Person.objects(name='Bob', age=30).upsert_one()
        assert 'Bob' == bob.name
        assert 30 == bob.age
        bob.name = 'Bobby'
        bob.save()
        bobby = self.Person.objects(name='Bobby', age=30).upsert_one()
        assert 'Bobby' == bobby.name
        assert 30 == bobby.age
        assert bob.id == bobby.id

    def test_set_on_insert(self):
        if False:
            i = 10
            return i + 15
        self.Person.drop_collection()
        self.Person.objects(pk=ObjectId()).update(set__name='Bob', set_on_insert__age=30, upsert=True)
        bob = self.Person.objects.first()
        assert 'Bob' == bob.name
        assert 30 == bob.age

    def test_rename(self):
        if False:
            return 10
        self.Person.drop_collection()
        self.Person.objects.create(name='Foo', age=11)
        bob = self.Person.objects.as_pymongo().first()
        assert 'age' in bob
        assert bob['age'] == 11
        self.Person.objects(name='Foo').update(rename__age='person_age')
        bob = self.Person.objects.as_pymongo().first()
        assert 'age' not in bob
        assert 'person_age' in bob
        assert bob['person_age'] == 11

    def test_save_and_only_on_fields_with_default(self):
        if False:
            print('Hello World!')

        class Embed(EmbeddedDocument):
            field = IntField()

        class B(Document):
            meta = {'collection': 'b'}
            field = IntField(default=1)
            embed = EmbeddedDocumentField(Embed, default=Embed)
            embed_no_default = EmbeddedDocumentField(Embed)
        val = 2
        embed = Embed()
        embed.field = val
        record = B()
        record.field = val
        record.embed = embed
        record.embed_no_default = embed
        record.save()
        record.reload()
        assert record.field == 2
        assert record.embed_no_default.field == 2
        assert record.embed.field == 2
        clone = B.objects().only('id').first()
        clone.save()
        record.reload()
        assert record.field == 2
        assert record.embed_no_default.field == 2
        assert record.embed.field == 2

    def test_bulk_insert(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that bulk insert works'

        class Comment(EmbeddedDocument):
            name = StringField()

        class Post(EmbeddedDocument):
            comments = ListField(EmbeddedDocumentField(Comment))

        class Blog(Document):
            title = StringField(unique=True)
            tags = ListField(StringField())
            posts = ListField(EmbeddedDocumentField(Post))
        Blog.drop_collection()
        assert 0 == Blog.objects.count()
        comment1 = Comment(name='testa')
        comment2 = Comment(name='testb')
        post1 = Post(comments=[comment1, comment2])
        post2 = Post(comments=[comment2, comment2])
        blogs = [Blog(title='%s' % i, posts=[post1, post2]) for i in range(99)]
        with query_counter() as q:
            assert q == 0
            Blog.objects.insert(blogs, load_bulk=False)
            assert q == 1
        assert Blog.objects.count() == len(blogs)
        Blog.drop_collection()
        Blog.ensure_indexes()
        blogs = [Blog(title='%s' % i, posts=[post1, post2]) for i in range(99)]
        with query_counter() as q:
            assert q == 0
            Blog.objects.insert(blogs)
            assert q == 2
        Blog.drop_collection()
        comment1 = Comment(name='testa')
        comment2 = Comment(name='testb')
        post1 = Post(comments=[comment1, comment2])
        post2 = Post(comments=[comment2, comment2])
        blog1 = Blog(title='code', posts=[post1, post2])
        blog2 = Blog(title='mongodb', posts=[post2, post1])
        (blog1, blog2) = Blog.objects.insert([blog1, blog2])
        assert blog1.title == 'code'
        assert blog2.title == 'mongodb'
        assert Blog.objects.count() == 2
        with pytest.raises(OperationError) as exc_info:
            blog = Blog.objects.first()
            Blog.objects.insert(blog)
        assert str(exc_info.value) == 'Some documents have ObjectIds, use doc.update() instead'
        with pytest.raises(OperationError) as exc_info:
            blogs_qs = Blog.objects
            Blog.objects.insert(blogs_qs)
        assert str(exc_info.value) == 'Some documents have ObjectIds, use doc.update() instead'
        new_post = Blog(title='code123', id=ObjectId())
        Blog.objects.insert(new_post)
        Blog.drop_collection()
        blog1 = Blog(title='code', posts=[post1, post2])
        blog1 = Blog.objects.insert(blog1)
        assert blog1.title == 'code'
        assert Blog.objects.count() == 1
        Blog.drop_collection()
        blog1 = Blog(title='code', posts=[post1, post2])
        obj_id = Blog.objects.insert(blog1, load_bulk=False)
        assert isinstance(obj_id, ObjectId)
        Blog.drop_collection()
        post3 = Post(comments=[comment1, comment1])
        blog1 = Blog(title='foo', posts=[post1, post2])
        blog2 = Blog(title='bar', posts=[post2, post3])
        Blog.objects.insert([blog1, blog2])
        with pytest.raises(NotUniqueError):
            Blog.objects.insert(Blog(title=blog2.title))
        assert Blog.objects.count() == 2

    def test_bulk_insert_different_class_fails(self):
        if False:
            return 10

        class Blog(Document):
            pass

        class Author(Document):
            pass
        with pytest.raises(OperationError):
            Blog.objects.insert(Author())

    def test_bulk_insert_with_wrong_type(self):
        if False:
            i = 10
            return i + 15

        class Blog(Document):
            name = StringField()
        Blog.drop_collection()
        Blog(name='test').save()
        with pytest.raises(OperationError):
            Blog.objects.insert('HELLO WORLD')
        with pytest.raises(OperationError):
            Blog.objects.insert({'name': 'garbage'})

    def test_bulk_insert_update_input_document_ids(self):
        if False:
            while True:
                i = 10

        class Comment(Document):
            idx = IntField()
        Comment.drop_collection()
        comments = [Comment(idx=idx) for idx in range(20)]
        for com in comments:
            assert com.id is None
        returned_comments = Comment.objects.insert(comments, load_bulk=True)
        for com in comments:
            assert isinstance(com.id, ObjectId)
        input_mapping = {com.id: com.idx for com in comments}
        saved_mapping = {com.id: com.idx for com in returned_comments}
        assert input_mapping == saved_mapping
        Comment.drop_collection()
        comment = Comment(idx=0)
        inserted_comment_id = Comment.objects.insert(comment, load_bulk=False)
        assert comment.id == inserted_comment_id

    def test_bulk_insert_accepts_doc_with_ids(self):
        if False:
            i = 10
            return i + 15

        class Comment(Document):
            id = IntField(primary_key=True)
        Comment.drop_collection()
        com1 = Comment(id=0)
        com2 = Comment(id=1)
        Comment.objects.insert([com1, com2])

    def test_insert_raise_if_duplicate_in_constraint(self):
        if False:
            return 10

        class Comment(Document):
            id = IntField(primary_key=True)
        Comment.drop_collection()
        com1 = Comment(id=0)
        Comment.objects.insert(com1)
        with pytest.raises(NotUniqueError):
            Comment.objects.insert(com1)

    def test_get_changed_fields_query_count(self):
        if False:
            return 10
        "Make sure we don't perform unnecessary db operations when\n        none of document's fields were updated.\n        "

        class Person(Document):
            name = StringField()
            owns = ListField(ReferenceField('Organization'))
            projects = ListField(ReferenceField('Project'))

        class Organization(Document):
            name = StringField()
            owner = ReferenceField(Person)
            employees = ListField(ReferenceField(Person))

        class Project(Document):
            name = StringField()
        Person.drop_collection()
        Organization.drop_collection()
        Project.drop_collection()
        r1 = Project(name='r1').save()
        r2 = Project(name='r2').save()
        r3 = Project(name='r3').save()
        p1 = Person(name='p1', projects=[r1, r2]).save()
        p2 = Person(name='p2', projects=[r2, r3]).save()
        o1 = Organization(name='o1', employees=[p1]).save()
        with query_counter() as q:
            assert q == 0
            org = Organization.objects.get(id=o1.id)
            assert q == 1
            org._get_changed_fields()
            assert q == 1
        org = Organization.objects.get(id=o1.id)
        with query_counter() as q:
            org.save()
            assert q == 0
        org = Organization.objects.get(id=o1.id)
        with query_counter() as q:
            org.save(cascade=False)
            assert q == 0
        org = Organization.objects.get(id=o1.id)
        with query_counter() as q:
            org.employees.append(p2)
            org.save()
            assert q == 2

    def test_repeated_iteration(self):
        if False:
            return 10
        'Ensure that QuerySet rewinds itself one iteration finishes.'
        self.Person(name='Person 1').save()
        self.Person(name='Person 2').save()
        queryset = self.Person.objects
        people1 = [person for person in queryset]
        people2 = [person for person in queryset]
        for _person in queryset:
            break
        people3 = [person for person in queryset]
        assert people1 == people2
        assert people1 == people3

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        'Test repr behavior isnt destructive'

        class Doc(Document):
            number = IntField()

            def __repr__(self):
                if False:
                    print('Hello World!')
                return '<Doc: %s>' % self.number
        Doc.drop_collection()
        for i in range(1000):
            Doc(number=i).save()
        docs = Doc.objects.order_by('number')
        assert docs.count() == 1000
        docs_string = '%s' % docs
        assert 'Doc: 0' in docs_string
        assert docs.count() == 1000
        assert '(remaining elements truncated)' in '%s' % docs
        docs = docs[1:4]
        assert '[<Doc: 1>, <Doc: 2>, <Doc: 3>]' == '%s' % docs
        assert docs.count(with_limit_and_skip=True) == 3
        for _ in docs:
            assert '.. queryset mid-iteration ..' == repr(docs)

    def test_regex_query_shortcuts(self):
        if False:
            while True:
                i = 10
        'Ensure that contains, startswith, endswith, etc work.'
        person = self.Person(name='Guido van Rossum')
        person.save()
        obj = self.Person.objects(name__contains='van').first()
        assert obj == person
        obj = self.Person.objects(name__contains='Van').first()
        assert obj is None
        obj = self.Person.objects(name__icontains='Van').first()
        assert obj == person
        obj = self.Person.objects(name__startswith='Guido').first()
        assert obj == person
        obj = self.Person.objects(name__startswith='guido').first()
        assert obj is None
        obj = self.Person.objects(name__istartswith='guido').first()
        assert obj == person
        obj = self.Person.objects(name__endswith='Rossum').first()
        assert obj == person
        obj = self.Person.objects(name__endswith='rossuM').first()
        assert obj is None
        obj = self.Person.objects(name__iendswith='rossuM').first()
        assert obj == person
        obj = self.Person.objects(name__exact='Guido van Rossum').first()
        assert obj == person
        obj = self.Person.objects(name__exact='Guido van rossum').first()
        assert obj is None
        obj = self.Person.objects(name__exact='Guido van Rossu').first()
        assert obj is None
        obj = self.Person.objects(name__iexact='gUIDO VAN rOSSUM').first()
        assert obj == person
        obj = self.Person.objects(name__iexact='gUIDO VAN rOSSU').first()
        assert obj is None
        obj = self.Person.objects(name__wholeword='Guido').first()
        assert obj == person
        obj = self.Person.objects(name__wholeword='rossum').first()
        assert obj is None
        obj = self.Person.objects(name__wholeword='Rossu').first()
        assert obj is None
        obj = self.Person.objects(name__iwholeword='rOSSUM').first()
        assert obj == person
        obj = self.Person.objects(name__iwholeword='rOSSU').first()
        assert obj is None
        obj = self.Person.objects(name__regex='^[Guido].*[Rossum]$').first()
        assert obj == person
        obj = self.Person.objects(name__regex='^[guido].*[rossum]$').first()
        assert obj is None
        obj = self.Person.objects(name__regex='^[uido].*[Rossum]$').first()
        assert obj is None
        obj = self.Person.objects(name__iregex='^[guido].*[rossum]$').first()
        assert obj == person
        obj = self.Person.objects(name__iregex='^[Uido].*[Rossum]$').first()
        assert obj is None
        person = self.Person(name="Guido van Rossum [.'Geek']")
        person.save()
        obj = self.Person.objects(name__icontains="[.'Geek").first()
        assert obj == person

    def test_not(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that the __not operator works as expected.'
        alice = self.Person(name='Alice', age=25)
        alice.save()
        obj = self.Person.objects(name__iexact='alice').first()
        assert obj == alice
        obj = self.Person.objects(name__not__iexact='alice').first()
        assert obj is None

    def test_filter_chaining(self):
        if False:
            print('Hello World!')
        'Ensure filters can be chained together.'

        class Blog(Document):
            id = StringField(primary_key=True)

        class BlogPost(Document):
            blog = ReferenceField(Blog)
            title = StringField()
            is_published = BooleanField()
            published_date = DateTimeField()

            @queryset_manager
            def published(doc_cls, queryset):
                if False:
                    while True:
                        i = 10
                return queryset(is_published=True)
        Blog.drop_collection()
        BlogPost.drop_collection()
        blog_1 = Blog(id='1')
        blog_2 = Blog(id='2')
        blog_3 = Blog(id='3')
        blog_1.save()
        blog_2.save()
        blog_3.save()
        BlogPost.objects.create(blog=blog_1, title='Blog Post #1', is_published=True, published_date=datetime.datetime(2010, 1, 5, 0, 0, 0))
        BlogPost.objects.create(blog=blog_2, title='Blog Post #2', is_published=True, published_date=datetime.datetime(2010, 1, 6, 0, 0, 0))
        BlogPost.objects.create(blog=blog_3, title='Blog Post #3', is_published=True, published_date=datetime.datetime(2010, 1, 7, 0, 0, 0))
        published_posts = BlogPost.published()
        published_posts = published_posts.filter(published_date__lt=datetime.datetime(2010, 1, 7, 0, 0, 0))
        assert published_posts.count() == 2
        blog_posts = BlogPost.objects
        blog_posts = blog_posts.filter(blog__in=[blog_1, blog_2])
        blog_posts = blog_posts.filter(blog=blog_3)
        assert blog_posts.count() == 0
        BlogPost.drop_collection()
        Blog.drop_collection()

    def test_filter_chaining_with_regex(self):
        if False:
            for i in range(10):
                print('nop')
        person = self.Person(name='Guido van Rossum')
        person.save()
        people = self.Person.objects
        people = people.filter(name__startswith='Gui').filter(name__not__endswith='tum').filter(name__icontains='VAN').filter(name__regex='^Guido').filter(name__wholeword='Guido').filter(name__wholeword='van')
        assert people.count() == 1

    def assertSequence(self, qs, expected):
        if False:
            print('Hello World!')
        qs = list(qs)
        expected = list(expected)
        assert len(qs) == len(expected)
        for i in range(len(qs)):
            assert qs[i] == expected[i]

    def test_ordering(self):
        if False:
            while True:
                i = 10
        'Ensure default ordering is applied and can be overridden.'

        class BlogPost(Document):
            title = StringField()
            published_date = DateTimeField()
            meta = {'ordering': ['-published_date']}
        BlogPost.drop_collection()
        blog_post_1 = BlogPost.objects.create(title='Blog Post #1', published_date=datetime.datetime(2010, 1, 5, 0, 0, 0))
        blog_post_2 = BlogPost.objects.create(title='Blog Post #2', published_date=datetime.datetime(2010, 1, 6, 0, 0, 0))
        blog_post_3 = BlogPost.objects.create(title='Blog Post #3', published_date=datetime.datetime(2010, 1, 7, 0, 0, 0))
        expected = [blog_post_3, blog_post_2, blog_post_1]
        self.assertSequence(BlogPost.objects.all(), expected)
        qs = BlogPost.objects.order_by('+published_date')
        expected = [blog_post_1, blog_post_2, blog_post_3]
        self.assertSequence(qs, expected)

    def test_clear_ordering(self):
        if False:
            print('Hello World!')
        'Ensure that the default ordering can be cleared by calling\n        order_by() w/o any arguments.\n        '
        (ORDER_BY_KEY, CMD_QUERY_KEY) = get_key_compat(self.mongodb_version)

        class BlogPost(Document):
            title = StringField()
            published_date = DateTimeField()
            meta = {'ordering': ['-published_date']}
        BlogPost.drop_collection()
        with db_ops_tracker() as q:
            BlogPost.objects.filter(title='whatever').first()
            assert len(q.get_ops()) == 1
            assert q.get_ops()[0][CMD_QUERY_KEY][ORDER_BY_KEY] == {'published_date': -1}
        with db_ops_tracker() as q:
            BlogPost.objects.filter(title='whatever').order_by().first()
            assert len(q.get_ops()) == 1
            assert ORDER_BY_KEY not in q.get_ops()[0][CMD_QUERY_KEY]
        with db_ops_tracker() as q:
            BlogPost.objects.filter(title='whatever').order_by('published_date').first()
            assert len(q.get_ops()) == 1
            assert q.get_ops()[0][CMD_QUERY_KEY][ORDER_BY_KEY] == {'published_date': 1}
        with db_ops_tracker() as q:
            qs = BlogPost.objects.filter(title='whatever').order_by('published_date')
            qs.order_by().first()
            assert len(q.get_ops()) == 1
            assert ORDER_BY_KEY not in q.get_ops()[0][CMD_QUERY_KEY]

    def test_no_ordering_for_get(self):
        if False:
            return 10
        "Ensure that Doc.objects.get doesn't use any ordering."
        (ORDER_BY_KEY, CMD_QUERY_KEY) = get_key_compat(self.mongodb_version)

        class BlogPost(Document):
            title = StringField()
            published_date = DateTimeField()
            meta = {'ordering': ['-published_date']}
        BlogPost.objects.create(title='whatever', published_date=datetime.datetime.utcnow())
        with db_ops_tracker() as q:
            BlogPost.objects.get(title='whatever')
            assert len(q.get_ops()) == 1
            assert ORDER_BY_KEY not in q.get_ops()[0][CMD_QUERY_KEY]
        with db_ops_tracker() as q:
            BlogPost.objects.order_by('-title').get(title='whatever')
            assert len(q.get_ops()) == 1
            assert ORDER_BY_KEY not in q.get_ops()[0][CMD_QUERY_KEY]

    def test_find_embedded(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that an embedded document is properly returned from\n        different manners of querying.\n        '

        class User(EmbeddedDocument):
            name = StringField()

        class BlogPost(Document):
            content = StringField()
            author = EmbeddedDocumentField(User)
        BlogPost.drop_collection()
        user = User(name='Test User')
        BlogPost.objects.create(author=user, content='Had a good coffee today...')
        result = BlogPost.objects.first()
        assert isinstance(result.author, User)
        assert result.author.name == 'Test User'
        result = BlogPost.objects.get(author__name=user.name)
        assert isinstance(result.author, User)
        assert result.author.name == 'Test User'
        result = BlogPost.objects.get(author={'name': user.name})
        assert isinstance(result.author, User)
        assert result.author.name == 'Test User'
        with pytest.raises(InvalidQueryError):
            BlogPost.objects.get(author=user.name)

    def test_find_empty_embedded(self):
        if False:
            print('Hello World!')
        'Ensure that you can save and find an empty embedded document.'

        class User(EmbeddedDocument):
            name = StringField()

        class BlogPost(Document):
            content = StringField()
            author = EmbeddedDocumentField(User)
        BlogPost.drop_collection()
        BlogPost.objects.create(content='Anonymous post...')
        result = BlogPost.objects.get(author=None)
        assert result.author is None

    def test_find_dict_item(self):
        if False:
            return 10
        'Ensure that DictField items may be found.'

        class BlogPost(Document):
            info = DictField()
        BlogPost.drop_collection()
        post = BlogPost(info={'title': 'test'})
        post.save()
        post_obj = BlogPost.objects(info__title='test').first()
        assert post_obj.id == post.id
        BlogPost.drop_collection()

    @requires_mongodb_lt_42
    def test_exec_js_query(self):
        if False:
            return 10
        'Ensure that queries are properly formed for use in exec_js.'

        class BlogPost(Document):
            hits = IntField()
            published = BooleanField()
        BlogPost.drop_collection()
        post1 = BlogPost(hits=1, published=False)
        post1.save()
        post2 = BlogPost(hits=1, published=True)
        post2.save()
        post3 = BlogPost(hits=1, published=True)
        post3.save()
        js_func = '\n            function(hitsField) {\n                var count = 0;\n                db[collection].find(query).forEach(function(doc) {\n                    count += doc[hitsField];\n                });\n                return count;\n            }\n        '
        c = BlogPost.objects(published=True).exec_js(js_func, 'hits')
        assert c == 2
        c = BlogPost.objects(published=False).exec_js(js_func, 'hits')
        assert c == 1
        BlogPost.drop_collection()

    @requires_mongodb_lt_42
    def test_exec_js_field_sub(self):
        if False:
            while True:
                i = 10
        'Ensure that field substitutions occur properly in exec_js functions.'

        class Comment(EmbeddedDocument):
            content = StringField(db_field='body')

        class BlogPost(Document):
            name = StringField(db_field='doc-name')
            comments = ListField(EmbeddedDocumentField(Comment), db_field='cmnts')
        BlogPost.drop_collection()
        comments1 = [Comment(content='cool'), Comment(content='yay')]
        post1 = BlogPost(name='post1', comments=comments1)
        post1.save()
        comments2 = [Comment(content='nice stuff')]
        post2 = BlogPost(name='post2', comments=comments2)
        post2.save()
        code = "\n        function getComments() {\n            var comments = [];\n            db[collection].find(query).forEach(function(doc) {\n                var docComments = doc[~comments];\n                for (var i = 0; i < docComments.length; i++) {\n                    comments.push({\n                        'document': doc[~name],\n                        'comment': doc[~comments][i][~comments.content]\n                    });\n                }\n            });\n            return comments;\n        }\n        "
        sub_code = BlogPost.objects._sub_js_fields(code)
        code_chunks = ['doc["cmnts"];', 'doc["doc-name"],', 'doc["cmnts"][i]["body"]']
        for chunk in code_chunks:
            assert chunk in sub_code
        results = BlogPost.objects.exec_js(code)
        expected_results = [{'comment': 'cool', 'document': 'post1'}, {'comment': 'yay', 'document': 'post1'}, {'comment': 'nice stuff', 'document': 'post2'}]
        assert results == expected_results
        code = '{{~comments.content}}'
        sub_code = BlogPost.objects._sub_js_fields(code)
        assert 'cmnts.body' == sub_code
        BlogPost.drop_collection()

    def test_delete(self):
        if False:
            print('Hello World!')
        'Ensure that documents are properly deleted from the database.'
        self.Person(name='User A', age=20).save()
        self.Person(name='User B', age=30).save()
        self.Person(name='User C', age=40).save()
        assert self.Person.objects.count() == 3
        self.Person.objects(age__lt=30).delete()
        assert self.Person.objects.count() == 2
        self.Person.objects.delete()
        assert self.Person.objects.count() == 0

    def test_reverse_delete_rule_cascade(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure cascading deletion of referring documents from the database.'

        class BlogPost(Document):
            content = StringField()
            author = ReferenceField(self.Person, reverse_delete_rule=CASCADE)
        BlogPost.drop_collection()
        me = self.Person(name='Test User')
        me.save()
        someoneelse = self.Person(name='Some-one Else')
        someoneelse.save()
        BlogPost(content='Watching TV', author=me).save()
        BlogPost(content='Chilling out', author=me).save()
        BlogPost(content='Pro Testing', author=someoneelse).save()
        assert 3 == BlogPost.objects.count()
        self.Person.objects(name='Test User').delete()
        assert 1 == BlogPost.objects.count()

    def test_reverse_delete_rule_cascade_on_abstract_document(self):
        if False:
            print('Hello World!')
        'Ensure cascading deletion of referring documents from the database\n        does not fail on abstract document.\n        '

        class AbstractBlogPost(Document):
            meta = {'abstract': True}
            author = ReferenceField(self.Person, reverse_delete_rule=CASCADE)

        class BlogPost(AbstractBlogPost):
            content = StringField()
        BlogPost.drop_collection()
        me = self.Person(name='Test User')
        me.save()
        someoneelse = self.Person(name='Some-one Else')
        someoneelse.save()
        BlogPost(content='Watching TV', author=me).save()
        BlogPost(content='Chilling out', author=me).save()
        BlogPost(content='Pro Testing', author=someoneelse).save()
        assert 3 == BlogPost.objects.count()
        self.Person.objects(name='Test User').delete()
        assert 1 == BlogPost.objects.count()

    def test_reverse_delete_rule_cascade_cycle(self):
        if False:
            while True:
                i = 10
        "Ensure reference cascading doesn't loop if reference graph isn't\n        a tree\n        "

        class Dummy(Document):
            reference = ReferenceField('self', reverse_delete_rule=CASCADE)
        base = Dummy().save()
        other = Dummy(reference=base).save()
        base.reference = other
        base.save()
        base.delete()
        with pytest.raises(DoesNotExist):
            base.reload()
        with pytest.raises(DoesNotExist):
            other.reload()

    def test_reverse_delete_rule_cascade_complex_cycle(self):
        if False:
            return 10
        "Ensure reference cascading doesn't loop if reference graph isn't\n        a tree\n        "

        class Category(Document):
            name = StringField()

        class Dummy(Document):
            reference = ReferenceField('self', reverse_delete_rule=CASCADE)
            cat = ReferenceField(Category, reverse_delete_rule=CASCADE)
        cat = Category(name='cat').save()
        base = Dummy(cat=cat).save()
        other = Dummy(reference=base).save()
        other2 = Dummy(reference=other).save()
        base.reference = other
        base.save()
        cat.delete()
        with pytest.raises(DoesNotExist):
            base.reload()
        with pytest.raises(DoesNotExist):
            other.reload()
        with pytest.raises(DoesNotExist):
            other2.reload()

    def test_reverse_delete_rule_cascade_self_referencing(self):
        if False:
            print('Hello World!')
        'Ensure self-referencing CASCADE deletes do not result in infinite\n        loop\n        '

        class Category(Document):
            name = StringField()
            parent = ReferenceField('self', reverse_delete_rule=CASCADE)
        Category.drop_collection()
        num_children = 3
        base = Category(name='Root')
        base.save()
        for i in range(num_children):
            child_name = 'Child-%i' % i
            child = Category(name=child_name, parent=base)
            child.save()
            for i in range(num_children):
                child_child_name = 'Child-Child-%i' % i
                child_child = Category(name=child_child_name, parent=child)
                child_child.save()
        tree_size = 1 + num_children + num_children * num_children
        assert tree_size == Category.objects.count()
        assert num_children == Category.objects(parent=base).count()
        base.delete()
        assert 0 == Category.objects.count()

    def test_reverse_delete_rule_nullify(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure nullification of references to deleted documents.'

        class Category(Document):
            name = StringField()

        class BlogPost(Document):
            content = StringField()
            category = ReferenceField(Category, reverse_delete_rule=NULLIFY)
        BlogPost.drop_collection()
        Category.drop_collection()
        lameness = Category(name='Lameness')
        lameness.save()
        post = BlogPost(content='Watching TV', category=lameness)
        post.save()
        assert BlogPost.objects.count() == 1
        assert BlogPost.objects.first().category.name == 'Lameness'
        Category.objects.delete()
        assert BlogPost.objects.count() == 1
        assert BlogPost.objects.first().category is None

    def test_reverse_delete_rule_nullify_on_abstract_document(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure nullification of references to deleted documents when\n        reference is on an abstract document.\n        '

        class AbstractBlogPost(Document):
            meta = {'abstract': True}
            author = ReferenceField(self.Person, reverse_delete_rule=NULLIFY)

        class BlogPost(AbstractBlogPost):
            content = StringField()
        BlogPost.drop_collection()
        me = self.Person(name='Test User')
        me.save()
        someoneelse = self.Person(name='Some-one Else')
        someoneelse.save()
        BlogPost(content='Watching TV', author=me).save()
        assert BlogPost.objects.count() == 1
        assert BlogPost.objects.first().author == me
        self.Person.objects(name='Test User').delete()
        assert BlogPost.objects.count() == 1
        assert BlogPost.objects.first().author is None

    def test_reverse_delete_rule_deny(self):
        if False:
            while True:
                i = 10
        'Ensure deletion gets denied on documents that still have references\n        to them.\n        '

        class BlogPost(Document):
            content = StringField()
            author = ReferenceField(self.Person, reverse_delete_rule=DENY)
        BlogPost.drop_collection()
        self.Person.drop_collection()
        me = self.Person(name='Test User')
        me.save()
        post = BlogPost(content='Watching TV', author=me)
        post.save()
        with pytest.raises(OperationError):
            self.Person.objects.delete()

    def test_reverse_delete_rule_deny_on_abstract_document(self):
        if False:
            i = 10
            return i + 15
        'Ensure deletion gets denied on documents that still have references\n        to them, when reference is on an abstract document.\n        '

        class AbstractBlogPost(Document):
            meta = {'abstract': True}
            author = ReferenceField(self.Person, reverse_delete_rule=DENY)

        class BlogPost(AbstractBlogPost):
            content = StringField()
        BlogPost.drop_collection()
        me = self.Person(name='Test User')
        me.save()
        BlogPost(content='Watching TV', author=me).save()
        assert 1 == BlogPost.objects.count()
        with pytest.raises(OperationError):
            self.Person.objects.delete()

    def test_reverse_delete_rule_pull(self):
        if False:
            print('Hello World!')
        'Ensure pulling of references to deleted documents.'

        class BlogPost(Document):
            content = StringField()
            authors = ListField(ReferenceField(self.Person, reverse_delete_rule=PULL))
        BlogPost.drop_collection()
        self.Person.drop_collection()
        me = self.Person(name='Test User')
        me.save()
        someoneelse = self.Person(name='Some-one Else')
        someoneelse.save()
        post = BlogPost(content='Watching TV', authors=[me, someoneelse])
        post.save()
        another = BlogPost(content='Chilling Out', authors=[someoneelse])
        another.save()
        someoneelse.delete()
        post.reload()
        another.reload()
        assert post.authors == [me]
        assert another.authors == []

    def test_reverse_delete_rule_pull_on_abstract_documents(self):
        if False:
            i = 10
            return i + 15
        'Ensure pulling of references to deleted documents when reference\n        is defined on an abstract document..\n        '

        class AbstractBlogPost(Document):
            meta = {'abstract': True}
            authors = ListField(ReferenceField(self.Person, reverse_delete_rule=PULL))

        class BlogPost(AbstractBlogPost):
            content = StringField()
        BlogPost.drop_collection()
        self.Person.drop_collection()
        me = self.Person(name='Test User')
        me.save()
        someoneelse = self.Person(name='Some-one Else')
        someoneelse.save()
        post = BlogPost(content='Watching TV', authors=[me, someoneelse])
        post.save()
        another = BlogPost(content='Chilling Out', authors=[someoneelse])
        another.save()
        someoneelse.delete()
        post.reload()
        another.reload()
        assert post.authors == [me]
        assert another.authors == []

    def test_delete_with_limits(self):
        if False:
            i = 10
            return i + 15

        class Log(Document):
            pass
        Log.drop_collection()
        for i in range(10):
            Log().save()
        Log.objects()[3:5].delete()
        assert 8 == Log.objects.count()

    def test_delete_with_limit_handles_delete_rules(self):
        if False:
            i = 10
            return i + 15
        'Ensure cascading deletion of referring documents from the database.'

        class BlogPost(Document):
            content = StringField()
            author = ReferenceField(self.Person, reverse_delete_rule=CASCADE)
        BlogPost.drop_collection()
        me = self.Person(name='Test User')
        me.save()
        someoneelse = self.Person(name='Some-one Else')
        someoneelse.save()
        BlogPost(content='Watching TV', author=me).save()
        BlogPost(content='Chilling out', author=me).save()
        BlogPost(content='Pro Testing', author=someoneelse).save()
        assert 3 == BlogPost.objects.count()
        self.Person.objects()[:1].delete()
        assert 1 == BlogPost.objects.count()

    def test_delete_edge_case_with_write_concern_0_return_None(self):
        if False:
            while True:
                i = 10
        "Return None if the delete operation is unacknowledged.\n\n        If we use an unack'd write concern, we don't really know how many\n        documents have been deleted.\n        "
        p1 = self.Person(name='User Z', age=20).save()
        del_result = p1.delete(w=0)
        assert del_result is None

    def test_reference_field_find(self):
        if False:
            return 10
        'Ensure cascading deletion of referring documents from the database.'

        class BlogPost(Document):
            content = StringField()
            author = ReferenceField(self.Person)
        BlogPost.drop_collection()
        self.Person.drop_collection()
        me = self.Person(name='Test User').save()
        BlogPost(content='test 123', author=me).save()
        assert 1 == BlogPost.objects(author=me).count()
        assert 1 == BlogPost.objects(author=me.pk).count()
        assert 1 == BlogPost.objects(author='%s' % me.pk).count()
        assert 1 == BlogPost.objects(author__in=[me]).count()
        assert 1 == BlogPost.objects(author__in=[me.pk]).count()
        assert 1 == BlogPost.objects(author__in=['%s' % me.pk]).count()

    def test_reference_field_find_dbref(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure cascading deletion of referring documents from the database.'

        class BlogPost(Document):
            content = StringField()
            author = ReferenceField(self.Person, dbref=True)
        BlogPost.drop_collection()
        self.Person.drop_collection()
        me = self.Person(name='Test User').save()
        BlogPost(content='test 123', author=me).save()
        assert 1 == BlogPost.objects(author=me).count()
        assert 1 == BlogPost.objects(author=me.pk).count()
        assert 1 == BlogPost.objects(author='%s' % me.pk).count()
        assert 1 == BlogPost.objects(author__in=[me]).count()
        assert 1 == BlogPost.objects(author__in=[me.pk]).count()
        assert 1 == BlogPost.objects(author__in=['%s' % me.pk]).count()

    def test_update_intfield_operator(self):
        if False:
            for i in range(10):
                print('nop')

        class BlogPost(Document):
            hits = IntField()
        BlogPost.drop_collection()
        post = BlogPost(hits=5)
        post.save()
        BlogPost.objects.update_one(set__hits=10)
        post.reload()
        assert post.hits == 10
        BlogPost.objects.update_one(inc__hits=1)
        post.reload()
        assert post.hits == 11
        BlogPost.objects.update_one(dec__hits=1)
        post.reload()
        assert post.hits == 10
        BlogPost.objects.update_one(dec__hits=-1)
        post.reload()
        assert post.hits == 11

    def test_update_decimalfield_operator(self):
        if False:
            for i in range(10):
                print('nop')

        class BlogPost(Document):
            review = DecimalField()
        BlogPost.drop_collection()
        post = BlogPost(review=3.5)
        post.save()
        BlogPost.objects.update_one(inc__review=0.1)
        post.reload()
        assert float(post.review) == 3.6
        BlogPost.objects.update_one(dec__review=0.1)
        post.reload()
        assert float(post.review) == 3.5
        BlogPost.objects.update_one(inc__review=Decimal(0.12))
        post.reload()
        assert float(post.review) == 3.62
        BlogPost.objects.update_one(dec__review=Decimal(0.12))
        post.reload()
        assert float(post.review) == 3.5

    def test_update_decimalfield_operator_not_working_with_force_string(self):
        if False:
            for i in range(10):
                print('nop')

        class BlogPost(Document):
            review = DecimalField(force_string=True)
        BlogPost.drop_collection()
        post = BlogPost(review=3.5)
        post.save()
        with pytest.raises(OperationError):
            BlogPost.objects.update_one(inc__review=0.1)

    def test_update_listfield_operator(self):
        if False:
            i = 10
            return i + 15
        'Ensure that atomic updates work properly.'

        class BlogPost(Document):
            tags = ListField(StringField())
        BlogPost.drop_collection()
        post = BlogPost(tags=['test'])
        post.save()
        BlogPost.objects.update(push__tags='mongo')
        post.reload()
        assert 'mongo' in post.tags
        BlogPost.objects.update_one(push_all__tags=['db', 'nosql'])
        post.reload()
        assert 'db' in post.tags
        assert 'nosql' in post.tags
        tags = post.tags[:-1]
        BlogPost.objects.update(pop__tags=1)
        post.reload()
        assert post.tags == tags
        BlogPost.objects.update_one(add_to_set__tags='unique')
        BlogPost.objects.update_one(add_to_set__tags='unique')
        post.reload()
        assert post.tags.count('unique') == 1
        BlogPost.drop_collection()

    def test_update_unset(self):
        if False:
            i = 10
            return i + 15

        class BlogPost(Document):
            title = StringField()
        BlogPost.drop_collection()
        post = BlogPost(title='garbage').save()
        assert post.title is not None
        BlogPost.objects.update_one(unset__title=1)
        post.reload()
        assert post.title is None
        pymongo_doc = BlogPost.objects.as_pymongo().first()
        assert 'title' not in pymongo_doc

    def test_update_push_with_position(self):
        if False:
            for i in range(10):
                print('nop')
        "Ensure that the 'push' update with position works properly."

        class BlogPost(Document):
            slug = StringField()
            tags = ListField(StringField())
        BlogPost.drop_collection()
        post = BlogPost.objects.create(slug='test')
        BlogPost.objects.filter(id=post.id).update(push__tags='code')
        BlogPost.objects.filter(id=post.id).update(push__tags__0=['mongodb', 'python'])
        post.reload()
        assert post.tags == ['mongodb', 'python', 'code']
        BlogPost.objects.filter(id=post.id).update(set__tags__2='java')
        post.reload()
        assert post.tags == ['mongodb', 'python', 'java']
        BlogPost.objects.filter(id=post.id).update(push__tags__0='scala')
        post.reload()
        assert post.tags == ['scala', 'mongodb', 'python', 'java']

    def test_update_push_list_of_list(self):
        if False:
            while True:
                i = 10
        "Ensure that the 'push' update operation works in the list of list"

        class BlogPost(Document):
            slug = StringField()
            tags = ListField()
        BlogPost.drop_collection()
        post = BlogPost(slug='test').save()
        BlogPost.objects.filter(slug='test').update(push__tags=['value1', 123])
        post.reload()
        assert post.tags == [['value1', 123]]

    def test_update_push_and_pull_add_to_set(self):
        if False:
            i = 10
            return i + 15
        "Ensure that the 'pull' update operation works correctly."

        class BlogPost(Document):
            slug = StringField()
            tags = ListField(StringField())
        BlogPost.drop_collection()
        post = BlogPost(slug='test')
        post.save()
        BlogPost.objects.filter(id=post.id).update(push__tags='code')
        post.reload()
        assert post.tags == ['code']
        BlogPost.objects.filter(id=post.id).update(push_all__tags=['mongodb', 'code'])
        post.reload()
        assert post.tags == ['code', 'mongodb', 'code']
        BlogPost.objects(slug='test').update(pull__tags='code')
        post.reload()
        assert post.tags == ['mongodb']
        BlogPost.objects(slug='test').update(pull_all__tags=['mongodb', 'code'])
        post.reload()
        assert post.tags == []
        BlogPost.objects(slug='test').update(__raw__={'$addToSet': {'tags': {'$each': ['code', 'mongodb', 'code']}}})
        post.reload()
        assert post.tags == ['code', 'mongodb']

    @requires_mongodb_gte_42
    def test_aggregation_update(self):
        if False:
            while True:
                i = 10
        "Ensure that the 'aggregation_update' update works correctly."

        class BlogPost(Document):
            slug = StringField()
            tags = ListField(StringField())
        BlogPost.drop_collection()
        post = BlogPost(slug='test')
        post.save()
        BlogPost.objects(slug='test').update(__raw__=[{'$set': {'slug': {'$concat': ['$slug', ' ', '$slug']}}}])
        post.reload()
        assert post.slug == 'test test'
        BlogPost.objects(slug='test test').update(__raw__=[{'$set': {'slug': {'$concat': ['$slug', ' ', 'it']}}}, {'$set': {'slug': {'$concat': ['When', ' ', '$slug']}}}])
        post.reload()
        assert post.slug == 'When test test it'

    def test_add_to_set_each(self):
        if False:
            i = 10
            return i + 15

        class Item(Document):
            name = StringField(required=True)
            description = StringField(max_length=50)
            parents = ListField(ReferenceField('self'))
        Item.drop_collection()
        item = Item(name='test item').save()
        parent_1 = Item(name='parent 1').save()
        parent_2 = Item(name='parent 2').save()
        item.update(add_to_set__parents=[parent_1, parent_2, parent_1])
        item.reload()
        assert [parent_1, parent_2] == item.parents

    def test_pull_nested(self):
        if False:
            for i in range(10):
                print('nop')

        class Collaborator(EmbeddedDocument):
            user = StringField()

            def __unicode__(self):
                if False:
                    print('Hello World!')
                return '%s' % self.user

        class Site(Document):
            name = StringField(max_length=75, unique=True, required=True)
            collaborators = ListField(EmbeddedDocumentField(Collaborator))
        Site.drop_collection()
        c = Collaborator(user='Esteban')
        s = Site(name='test', collaborators=[c]).save()
        Site.objects(id=s.id).update_one(pull__collaborators__user='Esteban')
        assert Site.objects.first().collaborators == []
        with pytest.raises(InvalidQueryError):
            Site.objects(id=s.id).update_one(pull_all__collaborators__user=['Ross'])

    def test_pull_from_nested_embedded(self):
        if False:
            print('Hello World!')

        class User(EmbeddedDocument):
            name = StringField()

            def __unicode__(self):
                if False:
                    return 10
                return '%s' % self.name

        class Collaborator(EmbeddedDocument):
            helpful = ListField(EmbeddedDocumentField(User))
            unhelpful = ListField(EmbeddedDocumentField(User))

        class Site(Document):
            name = StringField(max_length=75, unique=True, required=True)
            collaborators = EmbeddedDocumentField(Collaborator)
        Site.drop_collection()
        c = User(name='Esteban')
        f = User(name='Frank')
        s = Site(name='test', collaborators=Collaborator(helpful=[c], unhelpful=[f])).save()
        Site.objects(id=s.id).update_one(pull__collaborators__helpful=c)
        assert Site.objects.first().collaborators['helpful'] == []
        Site.objects(id=s.id).update_one(pull__collaborators__unhelpful={'name': 'Frank'})
        assert Site.objects.first().collaborators['unhelpful'] == []
        with pytest.raises(InvalidQueryError):
            Site.objects(id=s.id).update_one(pull_all__collaborators__helpful__name=['Ross'])

    def test_pull_from_nested_embedded_using_in_nin(self):
        if False:
            i = 10
            return i + 15
        "Ensure that the 'pull' update operation works on embedded documents using 'in' and 'nin' operators."

        class User(EmbeddedDocument):
            name = StringField()

            def __unicode__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return '%s' % self.name

        class Collaborator(EmbeddedDocument):
            helpful = ListField(EmbeddedDocumentField(User))
            unhelpful = ListField(EmbeddedDocumentField(User))

        class Site(Document):
            name = StringField(max_length=75, unique=True, required=True)
            collaborators = EmbeddedDocumentField(Collaborator)
        Site.drop_collection()
        a = User(name='Esteban')
        b = User(name='Frank')
        x = User(name='Harry')
        y = User(name='John')
        s = Site(name='test', collaborators=Collaborator(helpful=[a, b], unhelpful=[x, y])).save()
        Site.objects(id=s.id).update_one(pull__collaborators__helpful__name__in=['Esteban'])
        assert Site.objects.first().collaborators['helpful'] == [b]
        Site.objects(id=s.id).update_one(pull__collaborators__unhelpful__name__nin=['John'])
        assert Site.objects.first().collaborators['unhelpful'] == [y]

    def test_pull_from_nested_mapfield(self):
        if False:
            while True:
                i = 10

        class Collaborator(EmbeddedDocument):
            user = StringField()

            def __unicode__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return '%s' % self.user

        class Site(Document):
            name = StringField(max_length=75, unique=True, required=True)
            collaborators = MapField(ListField(EmbeddedDocumentField(Collaborator)))
        Site.drop_collection()
        c = Collaborator(user='Esteban')
        f = Collaborator(user='Frank')
        s = Site(name='test', collaborators={'helpful': [c], 'unhelpful': [f]})
        s.save()
        Site.objects(id=s.id).update_one(pull__collaborators__helpful__user='Esteban')
        assert Site.objects.first().collaborators['helpful'] == []
        Site.objects(id=s.id).update_one(pull__collaborators__unhelpful={'user': 'Frank'})
        assert Site.objects.first().collaborators['unhelpful'] == []
        with pytest.raises(InvalidQueryError):
            Site.objects(id=s.id).update_one(pull_all__collaborators__helpful__user=['Ross'])

    def test_pull_in_genericembedded_field(self):
        if False:
            while True:
                i = 10

        class Foo(EmbeddedDocument):
            name = StringField()

        class Bar(Document):
            foos = ListField(GenericEmbeddedDocumentField(choices=[Foo]))
        Bar.drop_collection()
        foo = Foo(name='bar')
        bar = Bar(foos=[foo]).save()
        Bar.objects(id=bar.id).update(pull__foos=foo)
        bar.reload()
        assert len(bar.foos) == 0

    def test_update_one_check_return_with_full_result(self):
        if False:
            print('Hello World!')

        class BlogTag(Document):
            name = StringField(required=True)
        BlogTag.drop_collection()
        BlogTag(name='garbage').save()
        default_update = BlogTag.objects.update_one(name='new')
        assert default_update == 1
        full_result_update = BlogTag.objects.update_one(name='new', full_result=True)
        assert isinstance(full_result_update, UpdateResult)

    def test_update_one_pop_generic_reference(self):
        if False:
            i = 10
            return i + 15

        class BlogTag(Document):
            name = StringField(required=True)

        class BlogPost(Document):
            slug = StringField()
            tags = ListField(ReferenceField(BlogTag), required=True)
        BlogPost.drop_collection()
        BlogTag.drop_collection()
        tag_1 = BlogTag(name='code')
        tag_1.save()
        tag_2 = BlogTag(name='mongodb')
        tag_2.save()
        post = BlogPost(slug='test', tags=[tag_1])
        post.save()
        post = BlogPost(slug='test-2', tags=[tag_1, tag_2])
        post.save()
        assert len(post.tags) == 2
        BlogPost.objects(slug='test-2').update_one(pop__tags=-1)
        post.reload()
        assert len(post.tags) == 1
        BlogPost.drop_collection()
        BlogTag.drop_collection()

    def test_editting_embedded_objects(self):
        if False:
            print('Hello World!')

        class BlogTag(EmbeddedDocument):
            name = StringField(required=True)

        class BlogPost(Document):
            slug = StringField()
            tags = ListField(EmbeddedDocumentField(BlogTag), required=True)
        BlogPost.drop_collection()
        tag_1 = BlogTag(name='code')
        tag_2 = BlogTag(name='mongodb')
        post = BlogPost(slug='test', tags=[tag_1])
        post.save()
        post = BlogPost(slug='test-2', tags=[tag_1, tag_2])
        post.save()
        assert len(post.tags) == 2
        BlogPost.objects(slug='test-2').update_one(set__tags__0__name='python')
        post.reload()
        assert post.tags[0].name == 'python'
        BlogPost.objects(slug='test-2').update_one(pop__tags=-1)
        post.reload()
        assert len(post.tags) == 1
        BlogPost.drop_collection()

    def test_set_list_embedded_documents(self):
        if False:
            return 10

        class Author(EmbeddedDocument):
            name = StringField()

        class Message(Document):
            title = StringField()
            authors = ListField(EmbeddedDocumentField('Author'))
        Message.drop_collection()
        message = Message(title='hello', authors=[Author(name='Harry')])
        message.save()
        Message.objects(authors__name='Harry').update_one(set__authors__S=Author(name='Ross'))
        message = message.reload()
        assert message.authors[0].name == 'Ross'
        Message.objects(authors__name='Ross').update_one(set__authors=[Author(name='Harry'), Author(name='Ross'), Author(name='Adam')])
        message = message.reload()
        assert message.authors[0].name == 'Harry'
        assert message.authors[1].name == 'Ross'
        assert message.authors[2].name == 'Adam'

    def test_set_generic_embedded_documents(self):
        if False:
            i = 10
            return i + 15

        class Bar(EmbeddedDocument):
            name = StringField()

        class User(Document):
            username = StringField()
            bar = GenericEmbeddedDocumentField(choices=[Bar])
        User.drop_collection()
        User(username='abc').save()
        User.objects(username='abc').update(set__bar=Bar(name='test'), upsert=True)
        user = User.objects(username='abc').first()
        assert user.bar.name == 'test'

    def test_reload_embedded_docs_instance(self):
        if False:
            while True:
                i = 10

        class SubDoc(EmbeddedDocument):
            val = IntField()

        class Doc(Document):
            embedded = EmbeddedDocumentField(SubDoc)
        doc = Doc(embedded=SubDoc(val=0)).save()
        doc.reload()
        assert doc.pk == doc.embedded._instance.pk

    def test_reload_list_embedded_docs_instance(self):
        if False:
            while True:
                i = 10

        class SubDoc(EmbeddedDocument):
            val = IntField()

        class Doc(Document):
            embedded = ListField(EmbeddedDocumentField(SubDoc))
        doc = Doc(embedded=[SubDoc(val=0)]).save()
        doc.reload()
        assert doc.pk == doc.embedded[0]._instance.pk

    def test_order_by(self):
        if False:
            i = 10
            return i + 15
        'Ensure that QuerySets may be ordered.'
        self.Person(name='User B', age=40).save()
        self.Person(name='User A', age=20).save()
        self.Person(name='User C', age=30).save()
        names = [p.name for p in self.Person.objects.order_by('-age')]
        assert names == ['User B', 'User C', 'User A']
        names = [p.name for p in self.Person.objects.order_by('+age')]
        assert names == ['User A', 'User C', 'User B']
        names = [p.name for p in self.Person.objects.order_by('age')]
        assert names == ['User A', 'User C', 'User B']
        ages = [p.age for p in self.Person.objects.order_by('-name')]
        assert ages == [30, 40, 20]
        ages = [p.age for p in self.Person.objects.order_by()]
        assert ages == [40, 20, 30]
        ages = [p.age for p in self.Person.objects.order_by('')]
        assert ages == [40, 20, 30]

    def test_order_by_optional(self):
        if False:
            i = 10
            return i + 15

        class BlogPost(Document):
            title = StringField()
            published_date = DateTimeField(required=False)
        BlogPost.drop_collection()
        blog_post_3 = BlogPost.objects.create(title='Blog Post #3', published_date=datetime.datetime(2010, 1, 6, 0, 0, 0))
        blog_post_2 = BlogPost.objects.create(title='Blog Post #2', published_date=datetime.datetime(2010, 1, 5, 0, 0, 0))
        blog_post_4 = BlogPost.objects.create(title='Blog Post #4', published_date=datetime.datetime(2010, 1, 7, 0, 0, 0))
        blog_post_1 = BlogPost.objects.create(title='Blog Post #1', published_date=None)
        expected = [blog_post_1, blog_post_2, blog_post_3, blog_post_4]
        self.assertSequence(BlogPost.objects.order_by('published_date'), expected)
        self.assertSequence(BlogPost.objects.order_by('+published_date'), expected)
        expected.reverse()
        self.assertSequence(BlogPost.objects.order_by('-published_date'), expected)

    def test_order_by_list(self):
        if False:
            while True:
                i = 10

        class BlogPost(Document):
            title = StringField()
            published_date = DateTimeField(required=False)
        BlogPost.drop_collection()
        blog_post_1 = BlogPost.objects.create(title='A', published_date=datetime.datetime(2010, 1, 6, 0, 0, 0))
        blog_post_2 = BlogPost.objects.create(title='B', published_date=datetime.datetime(2010, 1, 6, 0, 0, 0))
        blog_post_3 = BlogPost.objects.create(title='C', published_date=datetime.datetime(2010, 1, 7, 0, 0, 0))
        qs = BlogPost.objects.order_by('published_date', 'title')
        expected = [blog_post_1, blog_post_2, blog_post_3]
        self.assertSequence(qs, expected)
        qs = BlogPost.objects.order_by('-published_date', '-title')
        expected.reverse()
        self.assertSequence(qs, expected)

    def test_order_by_chaining(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that an order_by query chains properly and allows .only()'
        self.Person(name='User B', age=40).save()
        self.Person(name='User A', age=20).save()
        self.Person(name='User C', age=30).save()
        only_age = self.Person.objects.order_by('-age').only('age')
        names = [p.name for p in only_age]
        ages = [p.age for p in only_age]
        assert names == [None, None, None]
        assert ages == [40, 30, 20]
        qs = self.Person.objects.all().order_by('-age')
        qs = qs.limit(10)
        ages = [p.age for p in qs]
        assert ages == [40, 30, 20]
        qs = self.Person.objects.all().limit(10)
        qs = qs.order_by('-age')
        ages = [p.age for p in qs]
        assert ages == [40, 30, 20]
        qs = self.Person.objects.all().skip(0)
        qs = qs.order_by('-age')
        ages = [p.age for p in qs]
        assert ages == [40, 30, 20]

    def test_confirm_order_by_reference_wont_work(self):
        if False:
            for i in range(10):
                print('nop')
        'Ordering by reference is not possible.  Use map / reduce.. or\n        denormalise'

        class Author(Document):
            author = ReferenceField(self.Person)
        Author.drop_collection()
        person_a = self.Person(name='User A', age=20)
        person_a.save()
        person_b = self.Person(name='User B', age=40)
        person_b.save()
        person_c = self.Person(name='User C', age=30)
        person_c.save()
        Author(author=person_a).save()
        Author(author=person_b).save()
        Author(author=person_c).save()
        names = [a.author.name for a in Author.objects.order_by('-author__age')]
        assert names == ['User A', 'User B', 'User C']

    def test_comment(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure adding a comment to the query gets added to the query'
        MONGO_VER = self.mongodb_version
        (_, CMD_QUERY_KEY) = get_key_compat(MONGO_VER)
        QUERY_KEY = 'filter'
        COMMENT_KEY = 'comment'

        class User(Document):
            age = IntField()
        with db_ops_tracker() as q:
            User.objects.filter(age__gte=18).comment('looking for an adult').first()
            User.objects.comment('looking for an adult').filter(age__gte=18).first()
            ops = q.get_ops()
            assert len(ops) == 2
            for op in ops:
                assert op[CMD_QUERY_KEY][QUERY_KEY] == {'age': {'$gte': 18}}
                assert op[CMD_QUERY_KEY][COMMENT_KEY] == 'looking for an adult'

    def test_map_reduce(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure map/reduce is both mapping and reducing.'

        class BlogPost(Document):
            title = StringField()
            tags = ListField(StringField(), db_field='post-tag-list')
        BlogPost.drop_collection()
        BlogPost(title='Post #1', tags=['music', 'film', 'print']).save()
        BlogPost(title='Post #2', tags=['music', 'film']).save()
        BlogPost(title='Post #3', tags=['film', 'photography']).save()
        map_f = '\n            function() {\n                this[~tags].forEach(function(tag) {\n                    emit(tag, 1);\n                });\n            }\n        '
        reduce_f = '\n            function(key, values) {\n                var total = 0;\n                for(var i=0; i<values.length; i++) {\n                    total += values[i];\n                }\n                return total;\n            }\n        '
        results = BlogPost.objects.map_reduce(map_f, reduce_f, 'myresults')
        results = list(results)
        assert len(results) == 4
        music = list(filter(lambda r: r.key == 'music', results))[0]
        assert music.value == 2
        film = list(filter(lambda r: r.key == 'film', results))[0]
        assert film.value == 3
        BlogPost.drop_collection()

    def test_map_reduce_with_custom_object_ids(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that QuerySet.map_reduce works properly with custom\n        primary keys.\n        '

        class BlogPost(Document):
            title = StringField(primary_key=True)
            tags = ListField(StringField())
        BlogPost.drop_collection()
        post1 = BlogPost(title='Post #1', tags=['mongodb', 'mongoengine'])
        post2 = BlogPost(title='Post #2', tags=['django', 'mongodb'])
        post3 = BlogPost(title='Post #3', tags=['hitchcock films'])
        post1.save()
        post2.save()
        post3.save()
        assert BlogPost._fields['title'].db_field == '_id'
        assert BlogPost._meta['id_field'] == 'title'
        map_f = '\n            function() {\n                emit(this._id, 1);\n            }\n        '
        reduce_f = '\n            function(key, values) {\n                var total = 0;\n                for(var i=0; i<values.length; i++) {\n                    total += values[i];\n                }\n                return total;\n            }\n        '
        results = BlogPost.objects.order_by('_id').map_reduce(map_f, reduce_f, 'myresults2')
        results = list(results)
        assert len(results) == 3
        assert results[0].object.id == post1.id
        assert results[1].object.id == post2.id
        assert results[2].object.id == post3.id
        BlogPost.drop_collection()

    def test_map_reduce_custom_output(self):
        if False:
            while True:
                i = 10
        '\n        Test map/reduce custom output\n        '

        class Family(Document):
            id = IntField(primary_key=True)
            log = StringField()

        class Person(Document):
            id = IntField(primary_key=True)
            name = StringField()
            age = IntField()
            family = ReferenceField(Family)
        Family.drop_collection()
        Person.drop_collection()
        f1 = Family(id=1, log='Trav 02 de Julho')
        f1.save()
        Person(id=1, family=f1, name='Wilson Jr', age=21).save()
        Person(id=2, family=f1, name='Wilson Father', age=45).save()
        Person(id=3, family=f1, name='Eliana Costa', age=40).save()
        Person(id=4, family=f1, name='Tayza Mariana', age=17).save()
        f2 = Family(id=2, log='Av prof frasc brunno')
        f2.save()
        Person(id=5, family=f2, name='Isabella Luanna', age=16).save()
        Person(id=6, family=f2, name='Sandra Mara', age=36).save()
        Person(id=7, family=f2, name='Igor Gabriel', age=10).save()
        f3 = Family(id=3, log='Av brazil')
        f3.save()
        Person(id=8, family=f3, name='Arthur WA', age=30).save()
        Person(id=9, family=f3, name='Paula Leonel', age=25).save()
        map_person = '\n            function () {\n                emit(this.family, {\n                     totalAge: this.age,\n                     persons: [{\n                        name: this.name,\n                        age: this.age\n                }]});\n            }\n        '
        map_family = '\n            function () {\n                emit(this._id, {\n                   totalAge: 0,\n                   persons: []\n                });\n            }\n        '
        reduce_f = '\n            function (key, values) {\n                var family = {persons: [], totalAge: 0};\n\n                values.forEach(function(value) {\n                    if (value.persons) {\n                        value.persons.forEach(function (person) {\n                            family.persons.push(person);\n                            family.totalAge += person.age;\n                        });\n                        family.persons.sort((a, b) => (a.age > b.age))\n                    }\n                });\n\n                return family;\n            }\n        '
        cursor = Family.objects.map_reduce(map_f=map_family, reduce_f=reduce_f, output={'replace': 'family_map', 'db_alias': 'test2'})
        next(cursor)
        results = Person.objects.map_reduce(map_f=map_person, reduce_f=reduce_f, output={'reduce': 'family_map', 'db_alias': 'test2'})
        results = list(results)
        collection = get_db('test2').family_map
        assert collection.find_one({'_id': 1}) == {'_id': 1, 'value': {'persons': [{'age': 17, 'name': 'Tayza Mariana'}, {'age': 21, 'name': 'Wilson Jr'}, {'age': 40, 'name': 'Eliana Costa'}, {'age': 45, 'name': 'Wilson Father'}], 'totalAge': 123}}
        assert collection.find_one({'_id': 2}) == {'_id': 2, 'value': {'persons': [{'age': 10, 'name': 'Igor Gabriel'}, {'age': 16, 'name': 'Isabella Luanna'}, {'age': 36, 'name': 'Sandra Mara'}], 'totalAge': 62}}
        assert collection.find_one({'_id': 3}) == {'_id': 3, 'value': {'persons': [{'age': 25, 'name': 'Paula Leonel'}, {'age': 30, 'name': 'Arthur WA'}], 'totalAge': 55}}

    def test_map_reduce_finalize(self):
        if False:
            print('Hello World!')
        'Ensure that map, reduce, and finalize run and introduce "scope"\n        by simulating "hotness" ranking with Reddit algorithm.\n        '
        from time import mktime

        class Link(Document):
            title = StringField(db_field='bpTitle')
            up_votes = IntField()
            down_votes = IntField()
            submitted = DateTimeField(db_field='sTime')
        Link.drop_collection()
        now = datetime.datetime.utcnow()
        Link(title="Google Buzz auto-followed a woman's abusive ex ...", up_votes=1079, down_votes=553, submitted=now - datetime.timedelta(hours=4)).save()
        Link(title='We did it! Barbie is a computer engineer.', up_votes=481, down_votes=124, submitted=now - datetime.timedelta(hours=2)).save()
        Link(title='This Is A Mosquito Getting Killed By A Laser', up_votes=1446, down_votes=530, submitted=now - datetime.timedelta(hours=13)).save()
        Link(title='Arabic flashcards land physics student in jail.', up_votes=215, down_votes=105, submitted=now - datetime.timedelta(hours=6)).save()
        Link(title='The Burger Lab: Presenting, the Flood Burger', up_votes=48, down_votes=17, submitted=now - datetime.timedelta(hours=5)).save()
        Link(title='How to see polarization with the naked eye', up_votes=74, down_votes=13, submitted=now - datetime.timedelta(hours=10)).save()
        map_f = '\n            function() {\n                emit(this[~id], {up_delta: this[~up_votes] - this[~down_votes],\n                                sub_date: this[~submitted].getTime() / 1000})\n            }\n        '
        reduce_f = "\n            function(key, values) {\n                data = values[0];\n\n                x = data.up_delta;\n\n                // calculate time diff between reddit epoch and submission\n                sec_since_epoch = data.sub_date - reddit_epoch;\n\n                // calculate 'Y'\n                if(x > 0) {\n                    y = 1;\n                } else if (x = 0) {\n                    y = 0;\n                } else {\n                    y = -1;\n                }\n\n                // calculate 'Z', the maximal value\n                if(Math.abs(x) >= 1) {\n                    z = Math.abs(x);\n                } else {\n                    z = 1;\n                }\n\n                return {x: x, y: y, z: z, t_s: sec_since_epoch};\n            }\n        "
        finalize_f = '\n            function(key, value) {\n                // f(sec_since_epoch,y,z) =\n                //                    log10(z) + ((y*sec_since_epoch) / 45000)\n                z_10 = Math.log(value.z) / Math.log(10);\n                weight = z_10 + ((value.y * value.t_s) / 45000);\n                return weight;\n            }\n        '
        reddit_epoch = mktime(datetime.datetime(2005, 12, 8, 7, 46, 43).timetuple())
        scope = {'reddit_epoch': reddit_epoch}
        results = Link.objects.order_by('-value')
        results = results.map_reduce(map_f, reduce_f, 'myresults', finalize_f=finalize_f, scope=scope)
        results = list(results)
        assert results[0].object.title.startswith('Google Buzz')
        assert results[-1].object.title.startswith('How to see')
        Link.drop_collection()

    def test_item_frequencies(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that item frequencies are properly generated from lists.'

        class BlogPost(Document):
            hits = IntField()
            tags = ListField(StringField(), db_field='blogTags')
        BlogPost.drop_collection()
        BlogPost(hits=1, tags=['music', 'film', 'actors', 'watch']).save()
        BlogPost(hits=2, tags=['music', 'watch']).save()
        BlogPost(hits=2, tags=['music', 'actors']).save()

        def test_assertions(f):
            if False:
                return 10
            f = {key: int(val) for (key, val) in f.items()}
            assert {'music', 'film', 'actors', 'watch'} == set(f.keys())
            assert f['music'] == 3
            assert f['actors'] == 2
            assert f['watch'] == 2
            assert f['film'] == 1
        exec_js = BlogPost.objects.item_frequencies('tags')
        map_reduce = BlogPost.objects.item_frequencies('tags', map_reduce=True)
        test_assertions(exec_js)
        test_assertions(map_reduce)

        def test_assertions(f):
            if False:
                i = 10
                return i + 15
            f = {key: int(val) for (key, val) in f.items()}
            assert {'music', 'actors', 'watch'} == set(f.keys())
            assert f['music'] == 2
            assert f['actors'] == 1
            assert f['watch'] == 1
        exec_js = BlogPost.objects(hits__gt=1).item_frequencies('tags')
        map_reduce = BlogPost.objects(hits__gt=1).item_frequencies('tags', map_reduce=True)
        test_assertions(exec_js)
        test_assertions(map_reduce)

        def test_assertions(f):
            if False:
                print('Hello World!')
            assert round(abs(f['music'] - 3.0 / 8.0), 7) == 0
            assert round(abs(f['actors'] - 2.0 / 8.0), 7) == 0
            assert round(abs(f['watch'] - 2.0 / 8.0), 7) == 0
            assert round(abs(f['film'] - 1.0 / 8.0), 7) == 0
        exec_js = BlogPost.objects.item_frequencies('tags', normalize=True)
        map_reduce = BlogPost.objects.item_frequencies('tags', normalize=True, map_reduce=True)
        test_assertions(exec_js)
        test_assertions(map_reduce)

        def test_assertions(f):
            if False:
                for i in range(10):
                    print('nop')
            assert {1, 2} == set(f.keys())
            assert f[1] == 1
            assert f[2] == 2
        exec_js = BlogPost.objects.item_frequencies('hits')
        map_reduce = BlogPost.objects.item_frequencies('hits', map_reduce=True)
        test_assertions(exec_js)
        test_assertions(map_reduce)
        BlogPost.drop_collection()

    def test_item_frequencies_on_embedded(self):
        if False:
            while True:
                i = 10
        'Ensure that item frequencies are properly generated from lists.'

        class Phone(EmbeddedDocument):
            number = StringField()

        class Person(Document):
            name = StringField()
            phone = EmbeddedDocumentField(Phone)
        Person.drop_collection()
        doc = Person(name='Guido')
        doc.phone = Phone(number='62-3331-1656')
        doc.save()
        doc = Person(name='Marr')
        doc.phone = Phone(number='62-3331-1656')
        doc.save()
        doc = Person(name='WP Junior')
        doc.phone = Phone(number='62-3332-1656')
        doc.save()

        def test_assertions(f):
            if False:
                return 10
            f = {key: int(val) for (key, val) in f.items()}
            assert {'62-3331-1656', '62-3332-1656'} == set(f.keys())
            assert f['62-3331-1656'] == 2
            assert f['62-3332-1656'] == 1
        exec_js = Person.objects.item_frequencies('phone.number')
        map_reduce = Person.objects.item_frequencies('phone.number', map_reduce=True)
        test_assertions(exec_js)
        test_assertions(map_reduce)

        def test_assertions(f):
            if False:
                while True:
                    i = 10
            f = {key: int(val) for (key, val) in f.items()}
            assert {'62-3331-1656'} == set(f.keys())
            assert f['62-3331-1656'] == 2
        exec_js = Person.objects(phone__number='62-3331-1656').item_frequencies('phone.number')
        map_reduce = Person.objects(phone__number='62-3331-1656').item_frequencies('phone.number', map_reduce=True)
        test_assertions(exec_js)
        test_assertions(map_reduce)

        def test_assertions(f):
            if False:
                while True:
                    i = 10
            assert f['62-3331-1656'] == 2.0 / 3.0
            assert f['62-3332-1656'] == 1.0 / 3.0
        exec_js = Person.objects.item_frequencies('phone.number', normalize=True)
        map_reduce = Person.objects.item_frequencies('phone.number', normalize=True, map_reduce=True)
        test_assertions(exec_js)
        test_assertions(map_reduce)

    def test_item_frequencies_null_values(self):
        if False:
            i = 10
            return i + 15

        class Person(Document):
            name = StringField()
            city = StringField()
        Person.drop_collection()
        Person(name='Wilson Snr', city='CRB').save()
        Person(name='Wilson Jr').save()
        freq = Person.objects.item_frequencies('city')
        assert freq == {'CRB': 1.0, None: 1.0}
        freq = Person.objects.item_frequencies('city', normalize=True)
        assert freq == {'CRB': 0.5, None: 0.5}
        freq = Person.objects.item_frequencies('city', map_reduce=True)
        assert freq == {'CRB': 1.0, None: 1.0}
        freq = Person.objects.item_frequencies('city', normalize=True, map_reduce=True)
        assert freq == {'CRB': 0.5, None: 0.5}

    @requires_mongodb_lt_42
    def test_item_frequencies_with_null_embedded(self):
        if False:
            for i in range(10):
                print('nop')

        class Data(EmbeddedDocument):
            name = StringField()

        class Extra(EmbeddedDocument):
            tag = StringField()

        class Person(Document):
            data = EmbeddedDocumentField(Data, required=True)
            extra = EmbeddedDocumentField(Extra)
        Person.drop_collection()
        p = Person()
        p.data = Data(name='Wilson Jr')
        p.save()
        p = Person()
        p.data = Data(name='Wesley')
        p.extra = Extra(tag='friend')
        p.save()
        ot = Person.objects.item_frequencies('extra.tag', map_reduce=False)
        assert ot == {None: 1.0, 'friend': 1.0}
        ot = Person.objects.item_frequencies('extra.tag', map_reduce=True)
        assert ot == {None: 1.0, 'friend': 1.0}

    @requires_mongodb_lt_42
    def test_item_frequencies_with_0_values(self):
        if False:
            print('Hello World!')

        class Test(Document):
            val = IntField()
        Test.drop_collection()
        t = Test()
        t.val = 0
        t.save()
        ot = Test.objects.item_frequencies('val', map_reduce=True)
        assert ot == {0: 1}
        ot = Test.objects.item_frequencies('val', map_reduce=False)
        assert ot == {0: 1}

    @requires_mongodb_lt_42
    def test_item_frequencies_with_False_values(self):
        if False:
            while True:
                i = 10

        class Test(Document):
            val = BooleanField()
        Test.drop_collection()
        t = Test()
        t.val = False
        t.save()
        ot = Test.objects.item_frequencies('val', map_reduce=True)
        assert ot == {False: 1}
        ot = Test.objects.item_frequencies('val', map_reduce=False)
        assert ot == {False: 1}

    @requires_mongodb_lt_42
    def test_item_frequencies_normalize(self):
        if False:
            while True:
                i = 10

        class Test(Document):
            val = IntField()
        Test.drop_collection()
        for _ in range(50):
            Test(val=1).save()
        for _ in range(20):
            Test(val=2).save()
        freqs = Test.objects.item_frequencies('val', map_reduce=False, normalize=True)
        assert freqs == {1: 50.0 / 70, 2: 20.0 / 70}
        freqs = Test.objects.item_frequencies('val', map_reduce=True, normalize=True)
        assert freqs == {1: 50.0 / 70, 2: 20.0 / 70}

    def test_average(self):
        if False:
            return 10
        'Ensure that field can be averaged correctly.'
        self.Person(name='person', age=0).save()
        assert int(self.Person.objects.average('age')) == 0
        ages = [23, 54, 12, 94, 27]
        for (i, age) in enumerate(ages):
            self.Person(name='test%s' % i, age=age).save()
        avg = float(sum(ages)) / (len(ages) + 1)
        assert round(abs(int(self.Person.objects.average('age')) - avg), 7) == 0
        self.Person(name='ageless person').save()
        assert int(self.Person.objects.average('age')) == avg
        self.Person(name='person meta', person_meta=self.PersonMeta(weight=0)).save()
        assert round(abs(int(self.Person.objects.average('person_meta.weight')) - 0), 7) == 0
        for (i, weight) in enumerate(ages):
            self.Person(name=f'test meta{i}', person_meta=self.PersonMeta(weight=weight)).save()
        assert round(abs(int(self.Person.objects.average('person_meta.weight')) - avg), 7) == 0
        self.Person(name='test meta none').save()
        assert int(self.Person.objects.average('person_meta.weight')) == avg
        over_50 = [a for a in ages if a >= 50]
        avg = float(sum(over_50)) / len(over_50)
        assert self.Person.objects.filter(age__gte=50).average('age') == avg

    def test_sum(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that field can be summed over correctly.'
        ages = [23, 54, 12, 94, 27]
        for (i, age) in enumerate(ages):
            self.Person(name='test%s' % i, age=age).save()
        assert self.Person.objects.sum('age') == sum(ages)
        self.Person(name='ageless person').save()
        assert self.Person.objects.sum('age') == sum(ages)
        for (i, age) in enumerate(ages):
            self.Person(name='test meta%s' % i, person_meta=self.PersonMeta(weight=age)).save()
        assert self.Person.objects.sum('person_meta.weight') == sum(ages)
        self.Person(name='weightless person').save()
        assert self.Person.objects.sum('age') == sum(ages)
        assert self.Person.objects.filter(age__gte=50).sum('age') == sum((a for a in ages if a >= 50))

    def test_sum_over_db_field(self):
        if False:
            return 10
        'Ensure that a field mapped to a db field with a different name\n        can be summed over correctly.\n        '

        class UserVisit(Document):
            num_visits = IntField(db_field='visits')
        UserVisit.drop_collection()
        UserVisit.objects.create(num_visits=10)
        UserVisit.objects.create(num_visits=5)
        assert UserVisit.objects.sum('num_visits') == 15

    def test_average_over_db_field(self):
        if False:
            print('Hello World!')
        'Ensure that a field mapped to a db field with a different name\n        can have its average computed correctly.\n        '

        class UserVisit(Document):
            num_visits = IntField(db_field='visits')
        UserVisit.drop_collection()
        UserVisit.objects.create(num_visits=20)
        UserVisit.objects.create(num_visits=10)
        assert UserVisit.objects.average('num_visits') == 15

    def test_embedded_average(self):
        if False:
            while True:
                i = 10

        class Pay(EmbeddedDocument):
            value = DecimalField()

        class Doc(Document):
            name = StringField()
            pay = EmbeddedDocumentField(Pay)
        Doc.drop_collection()
        Doc(name='Wilson Junior', pay=Pay(value=150)).save()
        Doc(name='Isabella Luanna', pay=Pay(value=530)).save()
        Doc(name='Tayza mariana', pay=Pay(value=165)).save()
        Doc(name='Eliana Costa', pay=Pay(value=115)).save()
        assert Doc.objects.average('pay.value') == 240

    def test_embedded_array_average(self):
        if False:
            while True:
                i = 10

        class Pay(EmbeddedDocument):
            values = ListField(DecimalField())

        class Doc(Document):
            name = StringField()
            pay = EmbeddedDocumentField(Pay)
        Doc.drop_collection()
        Doc(name='Wilson Junior', pay=Pay(values=[150, 100])).save()
        Doc(name='Isabella Luanna', pay=Pay(values=[530, 100])).save()
        Doc(name='Tayza mariana', pay=Pay(values=[165, 100])).save()
        Doc(name='Eliana Costa', pay=Pay(values=[115, 100])).save()
        assert Doc.objects.average('pay.values') == 170

    def test_array_average(self):
        if False:
            i = 10
            return i + 15

        class Doc(Document):
            values = ListField(DecimalField())
        Doc.drop_collection()
        Doc(values=[150, 100]).save()
        Doc(values=[530, 100]).save()
        Doc(values=[165, 100]).save()
        Doc(values=[115, 100]).save()
        assert Doc.objects.average('values') == 170

    def test_embedded_sum(self):
        if False:
            while True:
                i = 10

        class Pay(EmbeddedDocument):
            value = DecimalField()

        class Doc(Document):
            name = StringField()
            pay = EmbeddedDocumentField(Pay)
        Doc.drop_collection()
        Doc(name='Wilson Junior', pay=Pay(value=150)).save()
        Doc(name='Isabella Luanna', pay=Pay(value=530)).save()
        Doc(name='Tayza mariana', pay=Pay(value=165)).save()
        Doc(name='Eliana Costa', pay=Pay(value=115)).save()
        assert Doc.objects.sum('pay.value') == 960

    def test_embedded_array_sum(self):
        if False:
            return 10

        class Pay(EmbeddedDocument):
            values = ListField(DecimalField())

        class Doc(Document):
            name = StringField()
            pay = EmbeddedDocumentField(Pay)
        Doc.drop_collection()
        Doc(name='Wilson Junior', pay=Pay(values=[150, 100])).save()
        Doc(name='Isabella Luanna', pay=Pay(values=[530, 100])).save()
        Doc(name='Tayza mariana', pay=Pay(values=[165, 100])).save()
        Doc(name='Eliana Costa', pay=Pay(values=[115, 100])).save()
        assert Doc.objects.sum('pay.values') == 1360

    def test_array_sum(self):
        if False:
            for i in range(10):
                print('nop')

        class Doc(Document):
            values = ListField(DecimalField())
        Doc.drop_collection()
        Doc(values=[150, 100]).save()
        Doc(values=[530, 100]).save()
        Doc(values=[165, 100]).save()
        Doc(values=[115, 100]).save()
        assert Doc.objects.sum('values') == 1360

    def test_distinct(self):
        if False:
            print('Hello World!')
        'Ensure that the QuerySet.distinct method works.'
        self.Person(name='Mr Orange', age=20).save()
        self.Person(name='Mr White', age=20).save()
        self.Person(name='Mr Orange', age=30).save()
        self.Person(name='Mr Pink', age=30).save()
        assert set(self.Person.objects.distinct('name')) == {'Mr Orange', 'Mr White', 'Mr Pink'}
        assert set(self.Person.objects.distinct('age')) == {20, 30}
        assert set(self.Person.objects(age=30).distinct('name')) == {'Mr Orange', 'Mr Pink'}

    def test_distinct_handles_references(self):
        if False:
            while True:
                i = 10

        class Foo(Document):
            bar = ReferenceField('Bar')

        class Bar(Document):
            text = StringField()
        Bar.drop_collection()
        Foo.drop_collection()
        bar = Bar(text='hi')
        bar.save()
        foo = Foo(bar=bar)
        foo.save()
        assert Foo.objects.distinct('bar') == [bar]
        assert Foo.objects.no_dereference().distinct('bar') == [bar.pk]

    def test_base_queryset_iter_raise_not_implemented(self):
        if False:
            while True:
                i = 10

        class Tmp(Document):
            pass
        qs = BaseQuerySet(document=Tmp, collection=Tmp._get_collection())
        with pytest.raises(NotImplementedError):
            _ = list(qs)

    def test_search_text_raise_if_called_2_times(self):
        if False:
            print('Hello World!')

        class News(Document):
            title = StringField()
            content = StringField()
            is_active = BooleanField(default=True)
        News.drop_collection()
        with pytest.raises(OperationError):
            News.objects.search_text('t1', language='portuguese').search_text('t2', language='french')

    def test_text_indexes(self):
        if False:
            while True:
                i = 10

        class News(Document):
            title = StringField()
            content = StringField()
            is_active = BooleanField(default=True)
            meta = {'indexes': [{'fields': ['$title', '$content'], 'default_language': 'portuguese', 'weights': {'title': 10, 'content': 2}}]}
        News.drop_collection()
        info = News.objects._collection.index_information()
        assert 'title_text_content_text' in info
        assert 'textIndexVersion' in info['title_text_content_text']
        News(title='Neymar quebrou a vertebra', content='O Brasil sofre com a perda de Neymar').save()
        News(title='Brasil passa para as quartas de finais', content='Com o brasil nas quartas de finais teremos um jogo complicado com a alemanha').save()
        count = News.objects.search_text('neymar', language='portuguese').count()
        assert count == 1
        count = News.objects.search_text('brasil -neymar').count()
        assert count == 1
        News(title='As eleições no Brasil já estão em planejamento', content='A candidata dilma roussef já começa o teu planejamento', is_active=False).save()
        new = News.objects(is_active=False).search_text('dilma', language='pt').first()
        query = News.objects(is_active=False).search_text('dilma', language='pt')._query
        assert query == {'$text': {'$search': 'dilma', '$language': 'pt'}, 'is_active': False}
        assert not new.is_active
        assert 'dilma' in new.content
        assert 'planejamento' in new.title
        query = News.objects.search_text('candidata')
        assert query._search_text == 'candidata'
        new = query.first()
        assert isinstance(new.get_text_score(), float)
        query = News.objects.search_text('brasil').order_by('$text_score')
        assert query._search_text == 'brasil'
        assert query.count() == 3
        assert query._query == {'$text': {'$search': 'brasil'}}
        cursor_args = query._cursor_args
        cursor_args_fields = cursor_args['projection']
        assert cursor_args_fields == {'_text_score': {'$meta': 'textScore'}}
        text_scores = [i.get_text_score() for i in query]
        assert len(text_scores) == 3
        assert text_scores[0] > text_scores[1]
        assert text_scores[1] > text_scores[2]
        max_text_score = text_scores[0]
        item = News.objects.search_text('brasil').order_by('$text_score').first()
        assert item.get_text_score() == max_text_score

    def test_distinct_handles_references_to_alias(self):
        if False:
            i = 10
            return i + 15
        register_connection('testdb', 'mongoenginetest2')

        class Foo(Document):
            bar = ReferenceField('Bar')
            meta = {'db_alias': 'testdb'}

        class Bar(Document):
            text = StringField()
            meta = {'db_alias': 'testdb'}
        Bar.drop_collection()
        Foo.drop_collection()
        bar = Bar(text='hi')
        bar.save()
        foo = Foo(bar=bar)
        foo.save()
        assert Foo.objects.distinct('bar') == [bar]

    def test_distinct_handles_db_field(self):
        if False:
            i = 10
            return i + 15
        'Ensure that distinct resolves field name to db_field as expected.'

        class Product(Document):
            product_id = IntField(db_field='pid')
        Product.drop_collection()
        Product(product_id=1).save()
        Product(product_id=2).save()
        Product(product_id=1).save()
        assert set(Product.objects.distinct('product_id')) == {1, 2}
        assert set(Product.objects.distinct('pid')) == {1, 2}
        Product.drop_collection()

    def test_distinct_ListField_EmbeddedDocumentField(self):
        if False:
            return 10

        class Author(EmbeddedDocument):
            name = StringField()

        class Book(Document):
            title = StringField()
            authors = ListField(EmbeddedDocumentField(Author))
        Book.drop_collection()
        mark_twain = Author(name='Mark Twain')
        john_tolkien = Author(name='John Ronald Reuel Tolkien')
        Book.objects.create(title='Tom Sawyer', authors=[mark_twain])
        Book.objects.create(title='The Lord of the Rings', authors=[john_tolkien])
        Book.objects.create(title='The Stories', authors=[mark_twain, john_tolkien])
        authors = Book.objects.distinct('authors')
        authors_names = {author.name for author in authors}
        assert authors_names == {mark_twain.name, john_tolkien.name}

    def test_distinct_ListField_EmbeddedDocumentField_EmbeddedDocumentField(self):
        if False:
            return 10

        class Continent(EmbeddedDocument):
            continent_name = StringField()

        class Country(EmbeddedDocument):
            country_name = StringField()
            continent = EmbeddedDocumentField(Continent)

        class Author(EmbeddedDocument):
            name = StringField()
            country = EmbeddedDocumentField(Country)

        class Book(Document):
            title = StringField()
            authors = ListField(EmbeddedDocumentField(Author))
        Book.drop_collection()
        europe = Continent(continent_name='europe')
        asia = Continent(continent_name='asia')
        scotland = Country(country_name='Scotland', continent=europe)
        tibet = Country(country_name='Tibet', continent=asia)
        mark_twain = Author(name='Mark Twain', country=scotland)
        john_tolkien = Author(name='John Ronald Reuel Tolkien', country=tibet)
        Book.objects.create(title='Tom Sawyer', authors=[mark_twain])
        Book.objects.create(title='The Lord of the Rings', authors=[john_tolkien])
        Book.objects.create(title='The Stories', authors=[mark_twain, john_tolkien])
        country_list = Book.objects.distinct('authors.country')
        assert country_list == [scotland, tibet]
        continent_list = Book.objects.distinct('authors.country.continent')
        continent_list_names = {c.continent_name for c in continent_list}
        assert continent_list_names == {europe.continent_name, asia.continent_name}

    def test_distinct_ListField_ReferenceField(self):
        if False:
            return 10

        class Bar(Document):
            text = StringField()

        class Foo(Document):
            bar = ReferenceField('Bar')
            bar_lst = ListField(ReferenceField('Bar'))
        Bar.drop_collection()
        Foo.drop_collection()
        bar_1 = Bar(text='hi')
        bar_1.save()
        bar_2 = Bar(text='bye')
        bar_2.save()
        foo = Foo(bar=bar_1, bar_lst=[bar_1, bar_2])
        foo.save()
        assert Foo.objects.distinct('bar_lst') == [bar_1, bar_2]
        assert Foo.objects.no_dereference().distinct('bar_lst') == [bar_1.pk, bar_2.pk]

    def test_custom_manager(self):
        if False:
            i = 10
            return i + 15
        'Ensure that custom QuerySetManager instances work as expected.'

        class BlogPost(Document):
            tags = ListField(StringField())
            deleted = BooleanField(default=False)
            date = DateTimeField(default=datetime.datetime.now)

            @queryset_manager
            def objects(cls, qryset):
                if False:
                    return 10
                opts = {'deleted': False}
                return qryset(**opts)

            @queryset_manager
            def objects_1_arg(qryset):
                if False:
                    i = 10
                    return i + 15
                opts = {'deleted': False}
                return qryset(**opts)

            @queryset_manager
            def music_posts(doc_cls, queryset, deleted=False):
                if False:
                    for i in range(10):
                        print('nop')
                return queryset(tags='music', deleted=deleted).order_by('date')
        BlogPost.drop_collection()
        post1 = BlogPost(tags=['music', 'film']).save()
        post2 = BlogPost(tags=['music']).save()
        post3 = BlogPost(tags=['film', 'actors']).save()
        post4 = BlogPost(tags=['film', 'actors', 'music'], deleted=True).save()
        assert [p.id for p in BlogPost.objects()] == [post1.id, post2.id, post3.id]
        assert [p.id for p in BlogPost.objects_1_arg()] == [post1.id, post2.id, post3.id]
        assert [p.id for p in BlogPost.music_posts()] == [post1.id, post2.id]
        assert [p.id for p in BlogPost.music_posts(True)] == [post4.id]
        BlogPost.drop_collection()

    def test_custom_manager_overriding_objects_works(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(Document):
            bar = StringField(default='bar')
            active = BooleanField(default=False)

            @queryset_manager
            def objects(doc_cls, queryset):
                if False:
                    print('Hello World!')
                return queryset(active=True)

            @queryset_manager
            def with_inactive(doc_cls, queryset):
                if False:
                    i = 10
                    return i + 15
                return queryset(active=False)
        Foo.drop_collection()
        Foo(active=True).save()
        Foo(active=False).save()
        assert 1 == Foo.objects.count()
        assert 1 == Foo.with_inactive.count()
        Foo.with_inactive.first().delete()
        assert 0 == Foo.with_inactive.count()
        assert 1 == Foo.objects.count()

    def test_inherit_objects(self):
        if False:
            while True:
                i = 10

        class Foo(Document):
            meta = {'allow_inheritance': True}
            active = BooleanField(default=True)

            @queryset_manager
            def objects(klass, queryset):
                if False:
                    i = 10
                    return i + 15
                return queryset(active=True)

        class Bar(Foo):
            pass
        Bar.drop_collection()
        Bar.objects.create(active=False)
        assert 0 == Bar.objects.count()

    def test_inherit_objects_override(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(Document):
            meta = {'allow_inheritance': True}
            active = BooleanField(default=True)

            @queryset_manager
            def objects(klass, queryset):
                if False:
                    return 10
                return queryset(active=True)

        class Bar(Foo):

            @queryset_manager
            def objects(klass, queryset):
                if False:
                    print('Hello World!')
                return queryset(active=False)
        Bar.drop_collection()
        Bar.objects.create(active=False)
        assert 0 == Foo.objects.count()
        assert 1 == Bar.objects.count()

    def test_query_value_conversion(self):
        if False:
            print('Hello World!')
        'Ensure that query values are properly converted when necessary.'

        class BlogPost(Document):
            author = ReferenceField(self.Person)
        BlogPost.drop_collection()
        person = self.Person(name='test', age=30)
        person.save()
        post = BlogPost(author=person)
        post.save()
        post_obj = BlogPost.objects(author=person).first()
        assert post.id == post_obj.id
        post_obj = BlogPost.objects(author__in=[person]).first()
        assert post.id == post_obj.id
        BlogPost.drop_collection()

    def test_update_value_conversion(self):
        if False:
            print('Hello World!')
        'Ensure that values used in updates are converted before use.'

        class Group(Document):
            members = ListField(ReferenceField(self.Person))
        Group.drop_collection()
        user1 = self.Person(name='user1')
        user1.save()
        user2 = self.Person(name='user2')
        user2.save()
        group = Group()
        group.save()
        Group.objects(id=group.id).update(set__members=[user1, user2])
        group.reload()
        assert len(group.members) == 2
        assert group.members[0].name == user1.name
        assert group.members[1].name == user2.name
        Group.drop_collection()

    def test_bulk(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure bulk querying by object id returns a proper dict.'

        class BlogPost(Document):
            title = StringField()
        BlogPost.drop_collection()
        post_1 = BlogPost(title='Post #1')
        post_2 = BlogPost(title='Post #2')
        post_3 = BlogPost(title='Post #3')
        post_4 = BlogPost(title='Post #4')
        post_5 = BlogPost(title='Post #5')
        post_1.save()
        post_2.save()
        post_3.save()
        post_4.save()
        post_5.save()
        ids = [post_1.id, post_2.id, post_5.id]
        objects = BlogPost.objects.in_bulk(ids)
        assert len(objects) == 3
        assert post_1.id in objects
        assert post_2.id in objects
        assert post_5.id in objects
        assert objects[post_1.id].title == post_1.title
        assert objects[post_2.id].title == post_2.title
        assert objects[post_5.id].title == post_5.title
        objects = BlogPost.objects.as_pymongo().in_bulk(ids)
        assert len(objects) == 3
        assert isinstance(objects[post_1.id], dict)
        BlogPost.drop_collection()

    def tearDown(self):
        if False:
            return 10
        self.Person.drop_collection()

    def test_custom_querysets(self):
        if False:
            return 10
        'Ensure that custom QuerySet classes may be used.'

        class CustomQuerySet(QuerySet):

            def not_empty(self):
                if False:
                    i = 10
                    return i + 15
                return self.count() > 0

        class Post(Document):
            meta = {'queryset_class': CustomQuerySet}
        Post.drop_collection()
        assert isinstance(Post.objects, CustomQuerySet)
        assert not Post.objects.not_empty()
        Post().save()
        assert Post.objects.not_empty()
        Post.drop_collection()

    def test_custom_querysets_set_manager_directly(self):
        if False:
            print('Hello World!')
        'Ensure that custom QuerySet classes may be used.'

        class CustomQuerySet(QuerySet):

            def not_empty(self):
                if False:
                    print('Hello World!')
                return self.count() > 0

        class CustomQuerySetManager(QuerySetManager):
            queryset_class = CustomQuerySet

        class Post(Document):
            objects = CustomQuerySetManager()
        Post.drop_collection()
        assert isinstance(Post.objects, CustomQuerySet)
        assert not Post.objects.not_empty()
        Post().save()
        assert Post.objects.not_empty()
        Post.drop_collection()

    def test_custom_querysets_set_manager_methods(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that custom QuerySet classes methods may be used.'

        class CustomQuerySet(QuerySet):

            def delete(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                'Example of method when one want to change default behaviour of it'
                return 0

        class CustomQuerySetManager(QuerySetManager):
            queryset_class = CustomQuerySet

        class Post(Document):
            objects = CustomQuerySetManager()
        Post.drop_collection()
        assert isinstance(Post.objects, CustomQuerySet)
        assert Post.objects.delete() == 0
        post = Post()
        post.save()
        assert Post.objects.count() == 1
        post.delete()
        assert Post.objects.count() == 1
        Post.drop_collection()

    def test_custom_querysets_managers_directly(self):
        if False:
            while True:
                i = 10
        'Ensure that custom QuerySet classes may be used.'

        class CustomQuerySetManager(QuerySetManager):

            @staticmethod
            def get_queryset(doc_cls, queryset):
                if False:
                    i = 10
                    return i + 15
                return queryset(is_published=True)

        class Post(Document):
            is_published = BooleanField(default=False)
            published = CustomQuerySetManager()
        Post.drop_collection()
        Post().save()
        Post(is_published=True).save()
        assert Post.objects.count() == 2
        assert Post.published.count() == 1
        Post.drop_collection()

    def test_custom_querysets_inherited(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that custom QuerySet classes may be used.'

        class CustomQuerySet(QuerySet):

            def not_empty(self):
                if False:
                    return 10
                return self.count() > 0

        class Base(Document):
            meta = {'abstract': True, 'queryset_class': CustomQuerySet}

        class Post(Base):
            pass
        Post.drop_collection()
        assert isinstance(Post.objects, CustomQuerySet)
        assert not Post.objects.not_empty()
        Post().save()
        assert Post.objects.not_empty()
        Post.drop_collection()

    def test_custom_querysets_inherited_direct(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that custom QuerySet classes may be used.'

        class CustomQuerySet(QuerySet):

            def not_empty(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.count() > 0

        class CustomQuerySetManager(QuerySetManager):
            queryset_class = CustomQuerySet

        class Base(Document):
            meta = {'abstract': True}
            objects = CustomQuerySetManager()

        class Post(Base):
            pass
        Post.drop_collection()
        assert isinstance(Post.objects, CustomQuerySet)
        assert not Post.objects.not_empty()
        Post().save()
        assert Post.objects.not_empty()
        Post.drop_collection()

    def test_count_limit_and_skip(self):
        if False:
            i = 10
            return i + 15

        class Post(Document):
            title = StringField()
        Post.drop_collection()
        for i in range(10):
            Post(title='Post %s' % i).save()
        assert 5 == Post.objects.limit(5).skip(5).count(with_limit_and_skip=True)
        assert 10 == Post.objects.limit(5).skip(5).count(with_limit_and_skip=False)

    def test_count_and_none(self):
        if False:
            return 10
        'Test count works with None()'

        class MyDoc(Document):
            pass
        MyDoc.drop_collection()
        for i in range(0, 10):
            MyDoc().save()
        assert MyDoc.objects.count() == 10
        assert MyDoc.objects.none().count() == 0

    def test_count_list_embedded(self):
        if False:
            print('Hello World!')

        class B(EmbeddedDocument):
            c = StringField()

        class A(Document):
            b = ListField(EmbeddedDocumentField(B))
        assert A.objects(b=[{'c': 'c'}]).count() == 0

    def test_call_after_limits_set(self):
        if False:
            return 10
        'Ensure that re-filtering after slicing works'

        class Post(Document):
            title = StringField()
        Post.drop_collection()
        Post(title='Post 1').save()
        Post(title='Post 2').save()
        posts = Post.objects.all()[0:1]
        assert len(list(posts())) == 1
        Post.drop_collection()

    def test_order_then_filter(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that ordering still works after filtering.'

        class Number(Document):
            n = IntField()
        Number.drop_collection()
        n2 = Number.objects.create(n=2)
        n1 = Number.objects.create(n=1)
        assert list(Number.objects) == [n2, n1]
        assert list(Number.objects.order_by('n')) == [n1, n2]
        assert list(Number.objects.order_by('n').filter()) == [n1, n2]
        Number.drop_collection()

    def test_clone(self):
        if False:
            return 10
        'Ensure that cloning clones complex querysets'

        class Number(Document):
            n = IntField()
        Number.drop_collection()
        for i in range(1, 101):
            t = Number(n=i)
            t.save()
        test = Number.objects
        test2 = test.clone()
        assert test != test2
        assert test.count() == test2.count()
        test = test.filter(n__gt=11)
        test2 = test.clone()
        assert test != test2
        assert test.count() == test2.count()
        test = test.limit(10)
        test2 = test.clone()
        assert test != test2
        assert test.count() == test2.count()
        Number.drop_collection()

    def test_clone_retains_settings(self):
        if False:
            return 10
        'Ensure that cloning retains the read_preference and read_concern'

        class Number(Document):
            n = IntField()
        Number.drop_collection()
        qs = Number.objects
        qs_clone = qs.clone()
        assert qs._read_preference == qs_clone._read_preference
        assert qs._read_concern == qs_clone._read_concern
        qs = Number.objects.read_preference(ReadPreference.PRIMARY_PREFERRED)
        qs_clone = qs.clone()
        assert qs._read_preference == ReadPreference.PRIMARY_PREFERRED
        assert qs._read_preference == qs_clone._read_preference
        qs = Number.objects.read_concern({'level': 'majority'})
        qs_clone = qs.clone()
        assert qs._read_concern.document == {'level': 'majority'}
        assert qs._read_concern == qs_clone._read_concern
        Number.drop_collection()

    def test_using(self):
        if False:
            while True:
                i = 10
        'Ensure that switching databases for a queryset is possible'

        class Number2(Document):
            n = IntField()
        Number2.drop_collection()
        with switch_db(Number2, 'test2') as Number2:
            Number2.drop_collection()
        for i in range(1, 10):
            t = Number2(n=i)
            t.switch_db('test2')
            t.save()
        assert len(Number2.objects.using('test2')) == 9

    def test_unset_reference(self):
        if False:
            while True:
                i = 10

        class Comment(Document):
            text = StringField()

        class Post(Document):
            comment = ReferenceField(Comment)
        Comment.drop_collection()
        Post.drop_collection()
        comment = Comment.objects.create(text='test')
        post = Post.objects.create(comment=comment)
        assert post.comment == comment
        Post.objects.update(unset__comment=1)
        post.reload()
        assert post.comment is None
        Comment.drop_collection()
        Post.drop_collection()

    def test_order_works_with_custom_db_field_names(self):
        if False:
            while True:
                i = 10

        class Number(Document):
            n = IntField(db_field='number')
        Number.drop_collection()
        n2 = Number.objects.create(n=2)
        n1 = Number.objects.create(n=1)
        assert list(Number.objects) == [n2, n1]
        assert list(Number.objects.order_by('n')) == [n1, n2]
        Number.drop_collection()

    def test_order_works_with_primary(self):
        if False:
            while True:
                i = 10
        'Ensure that order_by and primary work.'

        class Number(Document):
            n = IntField(primary_key=True)
        Number.drop_collection()
        Number(n=1).save()
        Number(n=2).save()
        Number(n=3).save()
        numbers = [n.n for n in Number.objects.order_by('-n')]
        assert [3, 2, 1] == numbers
        numbers = [n.n for n in Number.objects.order_by('+n')]
        assert [1, 2, 3] == numbers
        Number.drop_collection()

    def test_create_index(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that manual creation of indexes works.'

        class Comment(Document):
            message = StringField()
            meta = {'allow_inheritance': True}
        Comment.create_index('message')
        info = Comment.objects._collection.index_information()
        info = [(value['key'], value.get('unique', False), value.get('sparse', False)) for (key, value) in info.items()]
        assert ([('_cls', 1), ('message', 1)], False, False) in info

    def test_where(self):
        if False:
            return 10
        'Ensure that where clauses work.'

        class IntPair(Document):
            fielda = IntField()
            fieldb = IntField()
        IntPair.drop_collection()
        a = IntPair(fielda=1, fieldb=1)
        b = IntPair(fielda=1, fieldb=2)
        c = IntPair(fielda=2, fieldb=1)
        a.save()
        b.save()
        c.save()
        query = IntPair.objects.where('this[~fielda] >= this[~fieldb]')
        assert 'this["fielda"] >= this["fieldb"]' == query._where_clause
        results = list(query)
        assert 2 == len(results)
        assert a in results
        assert c in results
        query = IntPair.objects.where('this[~fielda] == this[~fieldb]')
        results = list(query)
        assert 1 == len(results)
        assert a in results
        query = IntPair.objects.where('function() { return this[~fielda] >= this[~fieldb] }')
        assert 'function() { return this["fielda"] >= this["fieldb"] }' == query._where_clause
        results = list(query)
        assert 2 == len(results)
        assert a in results
        assert c in results
        with pytest.raises(TypeError):
            list(IntPair.objects.where(fielda__gte=3))

    def test_scalar(self):
        if False:
            print('Hello World!')

        class Organization(Document):
            name = StringField()

        class User(Document):
            name = StringField()
            organization = ObjectIdField()
        User.drop_collection()
        Organization.drop_collection()
        whitehouse = Organization(name='White House')
        whitehouse.save()
        User(name='Bob Dole', organization=whitehouse.id).save()
        user_orgs = set(User.objects.scalar('organization'))
        orgs = Organization.objects(id__in=user_orgs).scalar('name')
        assert list(orgs) == ['White House']
        orgs = Organization.objects.scalar('name').in_bulk(list(user_orgs))
        user_map = User.objects.scalar('name', 'organization')
        user_listing = [(user, orgs[org]) for (user, org) in user_map]
        assert [('Bob Dole', 'White House')] == user_listing

    def test_scalar_simple(self):
        if False:
            print('Hello World!')

        class TestDoc(Document):
            x = IntField()
            y = BooleanField()
        TestDoc.drop_collection()
        TestDoc(x=10, y=True).save()
        TestDoc(x=20, y=False).save()
        TestDoc(x=30, y=True).save()
        plist = list(TestDoc.objects.scalar('x', 'y'))
        assert len(plist) == 3
        assert plist[0] == (10, True)
        assert plist[1] == (20, False)
        assert plist[2] == (30, True)

        class UserDoc(Document):
            name = StringField()
            age = IntField()
        UserDoc.drop_collection()
        UserDoc(name='Wilson Jr', age=19).save()
        UserDoc(name='Wilson', age=43).save()
        UserDoc(name='Eliana', age=37).save()
        UserDoc(name='Tayza', age=15).save()
        ulist = list(UserDoc.objects.scalar('name', 'age'))
        assert ulist == [('Wilson Jr', 19), ('Wilson', 43), ('Eliana', 37), ('Tayza', 15)]
        ulist = list(UserDoc.objects.scalar('name').order_by('age'))
        assert ulist == ['Tayza', 'Wilson Jr', 'Eliana', 'Wilson']

    def test_scalar_embedded(self):
        if False:
            while True:
                i = 10

        class Profile(EmbeddedDocument):
            name = StringField()
            age = IntField()

        class Locale(EmbeddedDocument):
            city = StringField()
            country = StringField()

        class Person(Document):
            profile = EmbeddedDocumentField(Profile)
            locale = EmbeddedDocumentField(Locale)
        Person.drop_collection()
        Person(profile=Profile(name='Wilson Jr', age=19), locale=Locale(city='Corumba-GO', country='Brazil')).save()
        Person(profile=Profile(name='Gabriel Falcao', age=23), locale=Locale(city='New York', country='USA')).save()
        Person(profile=Profile(name='Lincoln de souza', age=28), locale=Locale(city='Belo Horizonte', country='Brazil')).save()
        Person(profile=Profile(name='Walter cruz', age=30), locale=Locale(city='Brasilia', country='Brazil')).save()
        assert list(Person.objects.order_by('profile__age').scalar('profile__name')) == ['Wilson Jr', 'Gabriel Falcao', 'Lincoln de souza', 'Walter cruz']
        ulist = list(Person.objects.order_by('locale.city').scalar('profile__name', 'profile__age', 'locale__city'))
        assert ulist == [('Lincoln de souza', 28, 'Belo Horizonte'), ('Walter cruz', 30, 'Brasilia'), ('Wilson Jr', 19, 'Corumba-GO'), ('Gabriel Falcao', 23, 'New York')]

    def test_scalar_decimal(self):
        if False:
            print('Hello World!')
        from decimal import Decimal

        class Person(Document):
            name = StringField()
            rating = DecimalField()
        Person.drop_collection()
        Person(name='Wilson Jr', rating=Decimal('1.0')).save()
        ulist = list(Person.objects.scalar('name', 'rating'))
        assert ulist == [('Wilson Jr', Decimal('1.0'))]

    def test_scalar_reference_field(self):
        if False:
            return 10

        class State(Document):
            name = StringField()

        class Person(Document):
            name = StringField()
            state = ReferenceField(State)
        State.drop_collection()
        Person.drop_collection()
        s1 = State(name='Goias')
        s1.save()
        Person(name='Wilson JR', state=s1).save()
        plist = list(Person.objects.scalar('name', 'state'))
        assert plist == [('Wilson JR', s1)]

    def test_scalar_generic_reference_field(self):
        if False:
            print('Hello World!')

        class State(Document):
            name = StringField()

        class Person(Document):
            name = StringField()
            state = GenericReferenceField()
        State.drop_collection()
        Person.drop_collection()
        s1 = State(name='Goias')
        s1.save()
        Person(name='Wilson JR', state=s1).save()
        plist = list(Person.objects.scalar('name', 'state'))
        assert plist == [('Wilson JR', s1)]

    def test_generic_reference_field_with_only_and_as_pymongo(self):
        if False:
            for i in range(10):
                print('nop')

        class TestPerson(Document):
            name = StringField()

        class TestActivity(Document):
            name = StringField()
            owner = GenericReferenceField()
        TestPerson.drop_collection()
        TestActivity.drop_collection()
        person = TestPerson(name='owner')
        person.save()
        a1 = TestActivity(name='a1', owner=person)
        a1.save()
        activity = TestActivity.objects(owner=person).scalar('id', 'owner').no_dereference().first()
        assert activity[0] == a1.pk
        assert activity[1]['_ref'] == DBRef('test_person', person.pk)
        activity = TestActivity.objects(owner=person).only('id', 'owner')[0]
        assert activity.pk == a1.pk
        assert activity.owner == person
        activity = TestActivity.objects(owner=person).only('id', 'owner').as_pymongo().first()
        assert activity['_id'] == a1.pk
        assert activity['owner']['_ref'], DBRef('test_person', person.pk)

    def test_scalar_db_field(self):
        if False:
            print('Hello World!')

        class TestDoc(Document):
            x = IntField()
            y = BooleanField()
        TestDoc.drop_collection()
        TestDoc(x=10, y=True).save()
        TestDoc(x=20, y=False).save()
        TestDoc(x=30, y=True).save()
        plist = list(TestDoc.objects.scalar('x', 'y'))
        assert len(plist) == 3
        assert plist[0] == (10, True)
        assert plist[1] == (20, False)
        assert plist[2] == (30, True)

    def test_scalar_primary_key(self):
        if False:
            i = 10
            return i + 15

        class SettingValue(Document):
            key = StringField(primary_key=True)
            value = StringField()
        SettingValue.drop_collection()
        s = SettingValue(key='test', value='test value')
        s.save()
        val = SettingValue.objects.scalar('key', 'value')
        assert list(val) == [('test', 'test value')]

    def test_scalar_cursor_behaviour(self):
        if False:
            print('Hello World!')
        'Ensure that a query returns a valid set of results.'
        person1 = self.Person(name='User A', age=20)
        person1.save()
        person2 = self.Person(name='User B', age=30)
        person2.save()
        people = self.Person.objects.scalar('name')
        assert people.count() == 2
        results = list(people)
        assert results[0] == 'User A'
        assert results[1] == 'User B'
        people = self.Person.objects(age=20).scalar('name')
        assert people.count() == 1
        person = next(people)
        assert person == 'User A'
        people = list(self.Person.objects.limit(1).scalar('name'))
        assert len(people) == 1
        assert people[0] == 'User A'
        people = list(self.Person.objects.skip(1).scalar('name'))
        assert len(people) == 1
        assert people[0] == 'User B'
        person3 = self.Person(name='User C', age=40)
        person3.save()
        people = list(self.Person.objects[:2].scalar('name'))
        assert len(people) == 2
        assert people[0] == 'User A'
        assert people[1] == 'User B'
        people = list(self.Person.objects[1:].scalar('name'))
        assert len(people) == 2
        assert people[0] == 'User B'
        assert people[1] == 'User C'
        people = list(self.Person.objects[1:2].scalar('name'))
        assert len(people) == 1
        assert people[0] == 'User B'
        people = self.Person.objects[1:1]
        people = people.scalar('name')
        assert len(people) == 0
        people = list(self.Person.objects.scalar('name')[80000:80001])
        assert len(people) == 0
        self.Person.objects.delete()
        for i in range(55):
            self.Person(name='A%s' % i, age=i).save()
        assert self.Person.objects.scalar('name').count() == 55
        assert 'A0' == '%s' % self.Person.objects.order_by('name').scalar('name').first()
        assert 'A0' == '%s' % self.Person.objects.scalar('name').order_by('name')[0]
        assert "['A1', 'A2']" == '%s' % self.Person.objects.order_by('age').scalar('name')[1:3]
        assert "['A51', 'A52']" == '%s' % self.Person.objects.order_by('age').scalar('name')[51:53]
        person = self.Person.objects.order_by('name').first()
        assert 'A0' == '%s' % self.Person.objects.scalar('name').with_id(person.id)
        pks = self.Person.objects.order_by('age').scalar('pk')[1:3]
        names = self.Person.objects.scalar('name').in_bulk(list(pks)).values()
        expected = "['A1', 'A2']"
        assert expected == '%s' % sorted(names)

    def test_fields(self):
        if False:
            return 10

        class Bar(EmbeddedDocument):
            v = StringField()
            z = StringField()

        class Foo(Document):
            x = StringField()
            y = IntField()
            items = EmbeddedDocumentListField(Bar)
        Foo.drop_collection()
        Foo(x='foo1', y=1).save()
        Foo(x='foo2', y=2, items=[]).save()
        Foo(x='foo3', y=3, items=[Bar(z='a', v='V')]).save()
        Foo(x='foo4', y=4, items=[Bar(z='a', v='V'), Bar(z='b', v='W'), Bar(z='b', v='X'), Bar(z='c', v='V')]).save()
        Foo(x='foo5', y=5, items=[Bar(z='b', v='X'), Bar(z='c', v='V'), Bar(z='d', v='V'), Bar(z='e', v='V')]).save()
        foos_with_x = list(Foo.objects.order_by('y').fields(x=1))
        assert all((o.x is not None for o in foos_with_x))
        foos_without_y = list(Foo.objects.order_by('y').fields(y=0))
        assert all((o.y is None for o in foos_without_y))
        foos_with_sliced_items = list(Foo.objects.order_by('y').fields(slice__items=1))
        assert foos_with_sliced_items[0].items == []
        assert foos_with_sliced_items[1].items == []
        assert len(foos_with_sliced_items[2].items) == 1
        assert foos_with_sliced_items[2].items[0].z == 'a'
        assert len(foos_with_sliced_items[3].items) == 1
        assert foos_with_sliced_items[3].items[0].z == 'a'
        assert len(foos_with_sliced_items[4].items) == 1
        assert foos_with_sliced_items[4].items[0].z == 'b'
        foos_with_elem_match_items = list(Foo.objects.order_by('y').fields(elemMatch__items={'z': 'b'}))
        assert foos_with_elem_match_items[0].items == []
        assert foos_with_elem_match_items[1].items == []
        assert foos_with_elem_match_items[2].items == []
        assert len(foos_with_elem_match_items[3].items) == 1
        assert foos_with_elem_match_items[3].items[0].z == 'b'
        assert foos_with_elem_match_items[3].items[0].v == 'W'
        assert len(foos_with_elem_match_items[4].items) == 1
        assert foos_with_elem_match_items[4].items[0].z == 'b'

    def test_elem_match(self):
        if False:
            while True:
                i = 10

        class Foo(EmbeddedDocument):
            shape = StringField()
            color = StringField()
            thick = BooleanField()
            meta = {'allow_inheritance': False}

        class Bar(Document):
            foo = ListField(EmbeddedDocumentField(Foo))
            meta = {'allow_inheritance': False}
        Bar.drop_collection()
        b1 = Bar(foo=[Foo(shape='square', color='purple', thick=False), Foo(shape='circle', color='red', thick=True)])
        b1.save()
        b2 = Bar(foo=[Foo(shape='square', color='red', thick=True), Foo(shape='circle', color='purple', thick=False)])
        b2.save()
        b3 = Bar(foo=[Foo(shape='square', thick=True), Foo(shape='circle', color='purple', thick=False)])
        b3.save()
        ak = list(Bar.objects(foo__match={'shape': 'square', 'color': 'purple'}))
        assert [b1] == ak
        ak = list(Bar.objects(foo__elemMatch={'shape': 'square', 'color': 'purple'}))
        assert [b1] == ak
        ak = list(Bar.objects(foo__match=Foo(shape='square', color='purple')))
        assert [b1] == ak
        ak = list(Bar.objects(foo__elemMatch={'shape': 'square', 'color__exists': True}))
        assert [b1, b2] == ak
        ak = list(Bar.objects(foo__match={'shape': 'square', 'color__exists': True}))
        assert [b1, b2] == ak
        ak = list(Bar.objects(foo__elemMatch={'shape': 'square', 'color__exists': False}))
        assert [b3] == ak
        ak = list(Bar.objects(foo__match={'shape': 'square', 'color__exists': False}))
        assert [b3] == ak

    def test_upsert_includes_cls(self):
        if False:
            for i in range(10):
                print('nop')
        'Upserts should include _cls information for inheritable classes'

        class Test(Document):
            test = StringField()
        Test.drop_collection()
        Test.objects(test='foo').update_one(upsert=True, set__test='foo')
        assert '_cls' not in Test._collection.find_one()

        class Test(Document):
            meta = {'allow_inheritance': True}
            test = StringField()
        Test.drop_collection()
        Test.objects(test='foo').update_one(upsert=True, set__test='foo')
        assert '_cls' in Test._collection.find_one()

    def test_update_upsert_looks_like_a_digit(self):
        if False:
            return 10

        class MyDoc(DynamicDocument):
            pass
        MyDoc.drop_collection()
        assert 1 == MyDoc.objects.update_one(upsert=True, inc__47=1)
        assert MyDoc.objects.get()['47'] == 1

    def test_dictfield_key_looks_like_a_digit(self):
        if False:
            return 10
        'Only should work with DictField even if they have numeric keys.'

        class MyDoc(Document):
            test = DictField()
        MyDoc.drop_collection()
        doc = MyDoc(test={'47': 1})
        doc.save()
        assert MyDoc.objects.only('test__47').get().test['47'] == 1

    def test_clear_cls_query(self):
        if False:
            print('Hello World!')

        class Parent(Document):
            name = StringField()
            meta = {'allow_inheritance': True}

        class Child(Parent):
            age = IntField()
        Parent.drop_collection()
        assert Parent.objects._query == {'_cls': {'$in': ('Parent', 'Parent.Child')}}
        assert Parent.objects.clear_cls_query()._query == {}
        assert Parent.objects._query == {'_cls': {'$in': ('Parent', 'Parent.Child')}}
        assert Parent.objects.filter(name='xyz').clear_cls_query()._query == {'name': 'xyz'}
        Parent.objects.create(name='foo')
        Child.objects.create(name='bar', age=1)
        assert Parent.objects.clear_cls_query().count() == 2
        assert Parent.objects.count() == 2
        assert Child.objects().count() == 1
        assert Child.objects.clear_cls_query().count() == 2

    def test_read_preference(self):
        if False:
            while True:
                i = 10

        class Bar(Document):
            txt = StringField()
            meta = {'indexes': ['txt']}
        Bar.drop_collection()
        bar = Bar.objects.create(txt='xyz')
        bars = list(Bar.objects.read_preference(ReadPreference.PRIMARY))
        assert bars == [bar]
        bars = Bar.objects.read_preference(ReadPreference.SECONDARY_PREFERRED)
        assert bars._read_preference == ReadPreference.SECONDARY_PREFERRED
        assert bars._cursor.collection.read_preference == ReadPreference.SECONDARY_PREFERRED
        with pytest.raises(TypeError):
            Bar.objects.read_preference('Primary')

        def assert_read_pref(qs, expected_read_pref):
            if False:
                for i in range(10):
                    print('nop')
            assert qs._read_preference == expected_read_pref
            assert qs._cursor.collection.read_preference == expected_read_pref
        bars = Bar.objects.skip(1).read_preference(ReadPreference.SECONDARY_PREFERRED)
        assert_read_pref(bars, ReadPreference.SECONDARY_PREFERRED)
        bars = Bar.objects.limit(1).read_preference(ReadPreference.SECONDARY_PREFERRED)
        assert_read_pref(bars, ReadPreference.SECONDARY_PREFERRED)
        bars = Bar.objects.order_by('txt').read_preference(ReadPreference.SECONDARY_PREFERRED)
        assert_read_pref(bars, ReadPreference.SECONDARY_PREFERRED)
        bars = Bar.objects.hint([('txt', 1)]).read_preference(ReadPreference.SECONDARY_PREFERRED)
        assert_read_pref(bars, ReadPreference.SECONDARY_PREFERRED)

    def test_read_concern(self):
        if False:
            return 10

        class Bar(Document):
            txt = StringField()
            meta = {'indexes': ['txt']}
        Bar.drop_collection()
        bar = Bar.objects.create(txt='xyz')
        bars = list(Bar.objects.read_concern(None))
        assert bars == [bar]
        bars = Bar.objects.read_concern({'level': 'local'})
        assert bars._read_concern.document == {'level': 'local'}
        assert bars._cursor.collection.read_concern.document == {'level': 'local'}
        with pytest.raises(TypeError):
            Bar.objects.read_concern('local')

        def assert_read_concern(qs, expected_read_concern):
            if False:
                print('Hello World!')
            assert qs._read_concern.document == expected_read_concern
            assert qs._cursor.collection.read_concern.document == expected_read_concern
        bars = Bar.objects.skip(1).read_concern({'level': 'local'})
        assert_read_concern(bars, {'level': 'local'})
        bars = Bar.objects.limit(1).read_concern({'level': 'local'})
        assert_read_concern(bars, {'level': 'local'})
        bars = Bar.objects.order_by('txt').read_concern({'level': 'local'})
        assert_read_concern(bars, {'level': 'local'})
        bars = Bar.objects.hint([('txt', 1)]).read_concern({'level': 'majority'})
        assert_read_concern(bars, {'level': 'majority'})

    def test_json_simple(self):
        if False:
            i = 10
            return i + 15

        class Embedded(EmbeddedDocument):
            string = StringField()

        class Doc(Document):
            string = StringField()
            embedded_field = EmbeddedDocumentField(Embedded)
        Doc.drop_collection()
        Doc(string='Hi', embedded_field=Embedded(string='Hi')).save()
        Doc(string='Bye', embedded_field=Embedded(string='Bye')).save()
        Doc().save()
        json_data = Doc.objects.to_json(sort_keys=True, separators=(',', ':'))
        doc_objects = list(Doc.objects)
        assert doc_objects == Doc.objects.from_json(json_data)

    def test_json_complex(self):
        if False:
            i = 10
            return i + 15

        class EmbeddedDoc(EmbeddedDocument):
            pass

        class Simple(Document):
            pass

        class Doc(Document):
            string_field = StringField(default='1')
            int_field = IntField(default=1)
            float_field = FloatField(default=1.1)
            boolean_field = BooleanField(default=True)
            datetime_field = DateTimeField(default=datetime.datetime.now)
            embedded_document_field = EmbeddedDocumentField(EmbeddedDoc, default=lambda : EmbeddedDoc())
            list_field = ListField(default=lambda : [1, 2, 3])
            dict_field = DictField(default=lambda : {'hello': 'world'})
            objectid_field = ObjectIdField(default=ObjectId)
            reference_field = ReferenceField(Simple, default=lambda : Simple().save())
            map_field = MapField(IntField(), default=lambda : {'simple': 1})
            decimal_field = DecimalField(default=1.0)
            complex_datetime_field = ComplexDateTimeField(default=datetime.datetime.now)
            url_field = URLField(default='http://mongoengine.org')
            dynamic_field = DynamicField(default=1)
            generic_reference_field = GenericReferenceField(default=lambda : Simple().save())
            sorted_list_field = SortedListField(IntField(), default=lambda : [1, 2, 3])
            email_field = EmailField(default='ross@example.com')
            geo_point_field = GeoPointField(default=lambda : [1, 2])
            sequence_field = SequenceField()
            uuid_field = UUIDField(default=uuid.uuid4)
            generic_embedded_document_field = GenericEmbeddedDocumentField(default=lambda : EmbeddedDoc())
        Simple.drop_collection()
        Doc.drop_collection()
        Doc().save()
        json_data = Doc.objects.to_json()
        doc_objects = list(Doc.objects)
        assert doc_objects == Doc.objects.from_json(json_data)

    def test_as_pymongo(self):
        if False:
            for i in range(10):
                print('nop')

        class LastLogin(EmbeddedDocument):
            location = StringField()
            ip = StringField()

        class User(Document):
            id = StringField(primary_key=True)
            name = StringField()
            age = IntField()
            price = DecimalField()
            last_login = EmbeddedDocumentField(LastLogin)
        User.drop_collection()
        User.objects.create(id='Bob', name='Bob Dole', age=89, price=Decimal('1.11'))
        User.objects.create(id='Barak', name='Barak Obama', age=51, price=Decimal('2.22'), last_login=LastLogin(location='White House', ip='104.107.108.116'))
        results = User.objects.as_pymongo()
        assert set(results[0].keys()) == {'_id', 'name', 'age', 'price'}
        assert set(results[1].keys()) == {'_id', 'name', 'age', 'price', 'last_login'}
        results = User.objects.only('id', 'name').as_pymongo()
        assert set(results[0].keys()) == {'_id', 'name'}
        users = User.objects.only('name', 'price').as_pymongo()
        results = list(users)
        assert isinstance(results[0], dict)
        assert isinstance(results[1], dict)
        assert results[0]['name'] == 'Bob Dole'
        assert results[0]['price'] == 1.11
        assert results[1]['name'] == 'Barak Obama'
        assert results[1]['price'] == 2.22
        users = User.objects.only('name', 'last_login').as_pymongo()
        results = list(users)
        assert isinstance(results[0], dict)
        assert isinstance(results[1], dict)
        assert results[0] == {'_id': 'Bob', 'name': 'Bob Dole'}
        assert results[1] == {'_id': 'Barak', 'name': 'Barak Obama', 'last_login': {'location': 'White House', 'ip': '104.107.108.116'}}

    def test_as_pymongo_returns_cls_attribute_when_using_inheritance(self):
        if False:
            while True:
                i = 10

        class User(Document):
            name = StringField()
            meta = {'allow_inheritance': True}
        User.drop_collection()
        user = User(name='Bob Dole').save()
        result = User.objects.as_pymongo().first()
        assert result == {'_cls': 'User', '_id': user.id, 'name': 'Bob Dole'}

    def test_as_pymongo_json_limit_fields(self):
        if False:
            i = 10
            return i + 15

        class User(Document):
            email = EmailField(unique=True, required=True)
            password_hash = StringField(db_field='password_hash', required=True)
            password_salt = StringField(db_field='password_salt', required=True)
        User.drop_collection()
        User(email='ross@example.com', password_salt='SomeSalt', password_hash='SomeHash').save()
        serialized_user = User.objects.exclude('password_salt', 'password_hash').as_pymongo()[0]
        assert {'_id', 'email'} == set(serialized_user.keys())
        serialized_user = User.objects.exclude('id', 'password_salt', 'password_hash').to_json()
        assert '[{"email": "ross@example.com"}]' == serialized_user
        serialized_user = User.objects.only('email').as_pymongo()[0]
        assert {'_id', 'email'} == set(serialized_user.keys())
        serialized_user = User.objects.exclude('password_salt').only('email').as_pymongo()[0]
        assert {'_id', 'email'} == set(serialized_user.keys())
        serialized_user = User.objects.exclude('password_salt', 'id').only('email').as_pymongo()[0]
        assert {'email'} == set(serialized_user.keys())
        serialized_user = User.objects.exclude('password_salt', 'id').only('email').to_json()
        assert '[{"email": "ross@example.com"}]' == serialized_user

    def test_only_after_count(self):
        if False:
            return 10
        'Test that only() works after count()'

        class User(Document):
            name = StringField()
            age = IntField()
            address = StringField()
        User.drop_collection()
        user = User(name='User', age=50, address='Moscow, Russia').save()
        user_queryset = User.objects(age=50)
        result = user_queryset.only('name', 'age').as_pymongo().first()
        assert result == {'_id': user.id, 'name': 'User', 'age': 50}
        result = user_queryset.count()
        assert result == 1
        result = user_queryset.only('name', 'age').as_pymongo().first()
        assert result == {'_id': user.id, 'name': 'User', 'age': 50}

    def test_no_dereference(self):
        if False:
            print('Hello World!')

        class Organization(Document):
            name = StringField()

        class User(Document):
            name = StringField()
            organization = ReferenceField(Organization)
        User.drop_collection()
        Organization.drop_collection()
        whitehouse = Organization(name='White House').save()
        User(name='Bob Dole', organization=whitehouse).save()
        qs = User.objects()
        qs_user = qs.first()
        assert isinstance(qs.first().organization, Organization)
        assert isinstance(qs.no_dereference().first().organization, DBRef)
        assert isinstance(qs_user.organization, Organization)
        assert isinstance(qs.first().organization, Organization)

    def test_no_dereference_internals(self):
        if False:
            return 10

        class Organization(Document):
            name = StringField()

        class User(Document):
            organization = ReferenceField(Organization)
        User.drop_collection()
        Organization.drop_collection()
        cls_organization_field = User.organization
        assert cls_organization_field._auto_dereference, True
        org = Organization(name='whatever').save()
        User(organization=org).save()
        qs_no_deref = User.objects().no_dereference()
        user_no_deref = qs_no_deref.first()
        assert not qs_no_deref._auto_dereference
        instance_org_field = user_no_deref._fields['organization']
        assert instance_org_field is not cls_organization_field
        assert not instance_org_field._auto_dereference
        assert isinstance(user_no_deref.organization, DBRef)
        assert cls_organization_field._auto_dereference, True

    def test_no_dereference_no_side_effect_on_existing_instance(self):
        if False:
            return 10

        class Organization(Document):
            name = StringField()

        class User(Document):
            organization = ReferenceField(Organization)
            organization_gen = GenericReferenceField()
        User.drop_collection()
        Organization.drop_collection()
        org = Organization(name='whatever').save()
        User(organization=org, organization_gen=org).save()
        qs = User.objects()
        user = qs.first()
        qs_no_deref = User.objects().no_dereference()
        user_no_deref = qs_no_deref.first()
        no_derf_org = user_no_deref.organization
        assert isinstance(no_derf_org, DBRef)
        assert isinstance(user.organization, Organization)
        no_derf_org_gen = user_no_deref.organization_gen
        assert isinstance(no_derf_org_gen, dict)
        assert isinstance(user.organization_gen, Organization)

    def test_no_dereference_embedded_doc(self):
        if False:
            return 10

        class User(Document):
            name = StringField()

        class Member(EmbeddedDocument):
            name = StringField()
            user = ReferenceField(User)

        class Organization(Document):
            name = StringField()
            members = ListField(EmbeddedDocumentField(Member))
            ceo = ReferenceField(User)
            member = EmbeddedDocumentField(Member)
            admins = ListField(ReferenceField(User))
        Organization.drop_collection()
        User.drop_collection()
        user = User(name='Flash')
        user.save()
        member = Member(name='Flash', user=user)
        company = Organization(name='Mongo Inc', ceo=user, member=member, admins=[user], members=[member])
        company.save()
        org = Organization.objects().no_dereference().first()
        assert id(org._fields['admins']) != id(Organization.admins)
        assert not org._fields['admins']._auto_dereference
        admin = org.admins[0]
        assert isinstance(admin, DBRef)
        assert isinstance(org.member.user, DBRef)
        assert isinstance(org.members[0].user, DBRef)

    def test_cached_queryset(self):
        if False:
            for i in range(10):
                print('nop')

        class Person(Document):
            name = StringField()
        Person.drop_collection()
        for i in range(100):
            Person(name='No: %s' % i).save()
        with query_counter() as q:
            assert q == 0
            people = Person.objects
            [x for x in people]
            assert 100 == len(people._result_cache)
            import platform
            if platform.python_implementation() != 'PyPy':
                assert people._len is None
            assert q == 1
            list(people)
            assert 100 == people._len
            assert q == 1
            people.count(with_limit_and_skip=True)
            assert q == 1

    def test_no_cached_queryset(self):
        if False:
            print('Hello World!')

        class Person(Document):
            name = StringField()
        Person.drop_collection()
        for i in range(100):
            Person(name='No: %s' % i).save()
        with query_counter() as q:
            assert q == 0
            people = Person.objects.no_cache()
            [x for x in people]
            assert q == 1
            list(people)
            assert q == 2
            people.count()
            assert q == 3

    def test_no_cached_queryset__repr__(self):
        if False:
            return 10

        class Person(Document):
            name = StringField()
        Person.drop_collection()
        qs = Person.objects.no_cache()
        assert repr(qs) == '[]'

    def test_no_cached_on_a_cached_queryset_raise_error(self):
        if False:
            i = 10
            return i + 15

        class Person(Document):
            name = StringField()
        Person.drop_collection()
        Person(name='a').save()
        qs = Person.objects()
        _ = list(qs)
        with pytest.raises(OperationError, match='QuerySet already cached'):
            qs.no_cache()

    def test_no_cached_queryset_no_cache_back_to_cache(self):
        if False:
            while True:
                i = 10

        class Person(Document):
            name = StringField()
        Person.drop_collection()
        qs = Person.objects()
        assert isinstance(qs, QuerySet)
        qs = qs.no_cache()
        assert isinstance(qs, QuerySetNoCache)
        qs = qs.cache()
        assert isinstance(qs, QuerySet)

    def test_cache_not_cloned(self):
        if False:
            while True:
                i = 10

        class User(Document):
            name = StringField()

            def __unicode__(self):
                if False:
                    while True:
                        i = 10
                return self.name
        User.drop_collection()
        User(name='Alice').save()
        User(name='Bob').save()
        users = User.objects.all().order_by('name')
        assert '%s' % users == '[<User: Alice>, <User: Bob>]'
        assert 2 == len(users._result_cache)
        users = users.filter(name='Bob')
        assert '%s' % users == '[<User: Bob>]'
        assert 1 == len(users._result_cache)

    def test_no_cache(self):
        if False:
            print('Hello World!')
        'Ensure you can add meta data to file'

        class Noddy(Document):
            fields = DictField()
        Noddy.drop_collection()
        for i in range(100):
            noddy = Noddy()
            for j in range(20):
                noddy.fields['key' + str(j)] = 'value ' + str(j)
            noddy.save()
        docs = Noddy.objects.no_cache()
        counter = len([1 for i in docs])
        assert counter == 100
        assert len(list(docs)) == 100
        with pytest.raises(TypeError):
            len(docs)
        with query_counter() as q:
            list(docs)
            assert q == 1
        with query_counter() as q:
            list(docs)
            assert q == 1

    def test_nested_queryset_iterator(self):
        if False:
            return 10
        names = ['Alice', 'Bob', 'Chuck', 'David', 'Eric', 'Francis', 'George']

        class User(Document):
            name = StringField()

            def __unicode__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.name
        User.drop_collection()
        for name in names:
            User(name=name).save()
        users = User.objects.all().order_by('name')
        outer_count = 0
        inner_count = 0
        inner_total_count = 0
        with query_counter() as q:
            assert q == 0
            assert users.count(with_limit_and_skip=True) == 7
            for (i, outer_user) in enumerate(users):
                assert outer_user.name == names[i]
                outer_count += 1
                inner_count = 0
                assert users.count(with_limit_and_skip=True) == 7
                for (j, inner_user) in enumerate(users):
                    assert inner_user.name == names[j]
                    inner_count += 1
                    inner_total_count += 1
                assert inner_count == 7
            assert outer_count == 7
            assert inner_total_count == 7 * 7
            assert q == 2

    def test_no_sub_classes(self):
        if False:
            return 10

        class A(Document):
            x = IntField()
            y = IntField()
            meta = {'allow_inheritance': True}

        class B(A):
            z = IntField()

        class C(B):
            zz = IntField()
        A.drop_collection()
        A(x=10, y=20).save()
        A(x=15, y=30).save()
        B(x=20, y=40).save()
        B(x=30, y=50).save()
        C(x=40, y=60).save()
        assert A.objects.no_sub_classes().count() == 2
        assert A.objects.count() == 5
        assert B.objects.no_sub_classes().count() == 2
        assert B.objects.count() == 3
        assert C.objects.no_sub_classes().count() == 1
        assert C.objects.count() == 1
        for obj in A.objects.no_sub_classes():
            assert obj.__class__ == A
        for obj in B.objects.no_sub_classes():
            assert obj.__class__ == B
        for obj in C.objects.no_sub_classes():
            assert obj.__class__ == C

    def test_query_generic_embedded_document(self):
        if False:
            print('Hello World!')
        'Ensure that querying sub field on generic_embedded_field works'

        class A(EmbeddedDocument):
            a_name = StringField()

        class B(EmbeddedDocument):
            b_name = StringField()

        class Doc(Document):
            document = GenericEmbeddedDocumentField(choices=(A, B))
        Doc.drop_collection()
        Doc(document=A(a_name='A doc')).save()
        Doc(document=B(b_name='B doc')).save()
        assert Doc.objects(__raw__={'document.a_name': 'A doc'}).count() == 1
        assert Doc.objects(__raw__={'document.b_name': 'B doc'}).count() == 1
        assert Doc.objects(document__a_name='A doc').count() == 1
        assert Doc.objects(document__b_name='B doc').count() == 1

    def test_query_reference_to_custom_pk_doc(self):
        if False:
            while True:
                i = 10

        class A(Document):
            id = StringField(primary_key=True)

        class B(Document):
            a = ReferenceField(A)
        A.drop_collection()
        B.drop_collection()
        a = A.objects.create(id='custom_id')
        B.objects.create(a=a)
        assert B.objects.count() == 1
        assert B.objects.get(a=a).a == a
        assert B.objects.get(a=a.id).a == a

    def test_cls_query_in_subclassed_docs(self):
        if False:
            return 10

        class Animal(Document):
            name = StringField()
            meta = {'allow_inheritance': True}

        class Dog(Animal):
            pass

        class Cat(Animal):
            pass
        assert Animal.objects(name='Charlie')._query == {'name': 'Charlie', '_cls': {'$in': ('Animal', 'Animal.Dog', 'Animal.Cat')}}
        assert Dog.objects(name='Charlie')._query == {'name': 'Charlie', '_cls': 'Animal.Dog'}
        assert Cat.objects(name='Charlie')._query == {'name': 'Charlie', '_cls': 'Animal.Cat'}

    def test_can_have_field_same_name_as_query_operator(self):
        if False:
            return 10

        class Size(Document):
            name = StringField()

        class Example(Document):
            size = ReferenceField(Size)
        Size.drop_collection()
        Example.drop_collection()
        instance_size = Size(name='Large').save()
        Example(size=instance_size).save()
        assert Example.objects(size=instance_size).count() == 1
        assert Example.objects(size__in=[instance_size]).count() == 1

    def test_cursor_in_an_if_stmt(self):
        if False:
            while True:
                i = 10

        class Test(Document):
            test_field = StringField()
        Test.drop_collection()
        queryset = Test.objects
        if queryset:
            raise AssertionError('Empty cursor returns True')
        test = Test()
        test.test_field = 'test'
        test.save()
        queryset = Test.objects
        if not test:
            raise AssertionError('Cursor has data and returned False')
        next(queryset)
        if not queryset:
            raise AssertionError('Cursor has data and it must returns True, even in the last item.')

    def test_bool_performance(self):
        if False:
            return 10

        class Person(Document):
            name = StringField()
        Person.drop_collection()
        for i in range(100):
            Person(name='No: %s' % i).save()
        with query_counter() as q:
            if Person.objects:
                pass
            assert q == 1
            op = q.db.system.profile.find({'ns': {'$ne': '%s.system.indexes' % q.db.name}})[0]
            assert op['nreturned'] == 1

    def test_bool_with_ordering(self):
        if False:
            i = 10
            return i + 15
        (ORDER_BY_KEY, CMD_QUERY_KEY) = get_key_compat(self.mongodb_version)

        class Person(Document):
            name = StringField()
        Person.drop_collection()
        Person(name='Test').save()
        qs = Person.objects.order_by('name')
        with query_counter() as q:
            if bool(qs):
                pass
            op = q.db.system.profile.find({'ns': {'$ne': '%s.system.indexes' % q.db.name}})[0]
            assert ORDER_BY_KEY not in op[CMD_QUERY_KEY]
        qs2 = Person.objects.order_by('name')
        with query_counter() as q:
            for x in qs2:
                pass
            op = q.db.system.profile.find({'ns': {'$ne': '%s.system.indexes' % q.db.name}})[0]
            assert ORDER_BY_KEY in op[CMD_QUERY_KEY]

    def test_bool_with_ordering_from_meta_dict(self):
        if False:
            for i in range(10):
                print('nop')
        (ORDER_BY_KEY, CMD_QUERY_KEY) = get_key_compat(self.mongodb_version)

        class Person(Document):
            name = StringField()
            meta = {'ordering': ['name']}
        Person.drop_collection()
        Person(name='B').save()
        Person(name='C').save()
        Person(name='A').save()
        with query_counter() as q:
            if Person.objects:
                pass
            op = q.db.system.profile.find({'ns': {'$ne': '%s.system.indexes' % q.db.name}})[0]
            assert '$orderby' not in op[CMD_QUERY_KEY], 'BaseQuerySet must remove orderby from meta in boolen test'
            assert Person.objects.first().name == 'A'
            assert Person.objects._has_data(), 'Cursor has data and returned False'

    def test_delete_count(self):
        if False:
            i = 10
            return i + 15
        [self.Person(name=f'User {i}', age=i * 10).save() for i in range(1, 4)]
        assert self.Person.objects().delete() == 3
        [self.Person(name=f'User {i}', age=i * 10).save() for i in range(1, 4)]
        assert self.Person.objects().skip(1).delete() == 2
        self.Person.objects().delete()
        assert self.Person.objects().skip(1).delete() == 0

    def test_max_time_ms(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError):
            self.Person.objects(name='name').max_time_ms('not a number')

    def test_subclass_field_query(self):
        if False:
            i = 10
            return i + 15

        class Animal(Document):
            is_mamal = BooleanField()
            meta = {'allow_inheritance': True}

        class Cat(Animal):
            whiskers_length = FloatField()

        class ScottishCat(Cat):
            folded_ears = BooleanField()
        Animal.drop_collection()
        Animal(is_mamal=False).save()
        Cat(is_mamal=True, whiskers_length=5.1).save()
        ScottishCat(is_mamal=True, folded_ears=True).save()
        assert Animal.objects(folded_ears=True).count() == 1
        assert Animal.objects(whiskers_length=5.1).count() == 1

    def test_loop_over_invalid_id_does_not_crash(self):
        if False:
            while True:
                i = 10

        class Person(Document):
            name = StringField()
        Person.drop_collection()
        Person._get_collection().insert_one({'name': 'a', 'id': ''})
        for p in Person.objects():
            assert p.name == 'a'

    def test_len_during_iteration(self):
        if False:
            print('Hello World!')
        "Tests that calling len on a queyset during iteration doesn't\n        stop paging.\n        "

        class Data(Document):
            pass
        for i in range(300):
            Data().save()
        records = Data.objects.limit(250)
        len(records)
        for (i, r) in enumerate(records):
            if i == 58:
                len(records)
        assert i == 249
        records = Data.objects.limit(250)
        for (i, r) in enumerate(records):
            if i == 58:
                len(records)
        assert i == 249

    def test_iteration_within_iteration(self):
        if False:
            for i in range(10):
                print('nop')
        'You should be able to reliably iterate over all the documents\n        in a given queryset even if there are multiple iterations of it\n        happening at the same time.\n        '

        class Data(Document):
            pass
        for i in range(300):
            Data().save()
        qs = Data.objects.limit(250)
        for (i, doc) in enumerate(qs):
            for (j, doc2) in enumerate(qs):
                pass
        assert i == 249
        assert j == 249

    def test_in_operator_on_non_iterable(self):
        if False:
            print('Hello World!')
        'Ensure that using the `__in` operator on a non-iterable raises an\n        error.\n        '

        class User(Document):
            name = StringField()

        class BlogPost(Document):
            content = StringField()
            authors = ListField(ReferenceField(User))
        User.drop_collection()
        BlogPost.drop_collection()
        author = User.objects.create(name='Test User')
        post = BlogPost.objects.create(content='Had a good coffee today...', authors=[author])
        blog_posts = BlogPost.objects(authors__in=[author])
        assert list(blog_posts) == [post]
        with pytest.raises(TypeError):
            BlogPost.objects(authors__in=author.pk).count()
        with pytest.raises(TypeError):
            BlogPost.objects(authors__in=author).count()

    def test_create_count(self):
        if False:
            i = 10
            return i + 15
        self.Person.drop_collection()
        self.Person.objects.create(name='Foo')
        self.Person.objects.create(name='Bar')
        self.Person.objects.create(name='Baz')
        assert self.Person.objects.count(with_limit_and_skip=True) == 3
        self.Person.objects.create(name='Foo_1')
        assert self.Person.objects.count(with_limit_and_skip=True) == 4

    def test_no_cursor_timeout(self):
        if False:
            i = 10
            return i + 15
        qs = self.Person.objects()
        assert qs._cursor_args == {}
        qs = self.Person.objects().timeout(True)
        assert qs._cursor_args == {}
        qs = self.Person.objects().timeout(False)
        assert qs._cursor_args == {'no_cursor_timeout': True}

    @requires_mongodb_gte_44
    def test_allow_disk_use(self):
        if False:
            for i in range(10):
                print('nop')
        qs = self.Person.objects()
        assert qs._cursor_args == {}
        qs = self.Person.objects().allow_disk_use(False)
        assert qs._cursor_args == {}
        qs = self.Person.objects().allow_disk_use(True)
        assert qs._cursor_args == {'allow_disk_use': True}
        self.Person.drop_collection()
        self.Person.objects.create(name='Foo', age=12)
        self.Person.objects.create(name='Baz', age=17)
        self.Person.objects.create(name='Bar', age=13)
        qs_disk = self.Person.objects().order_by('age').allow_disk_use(True)
        qs = self.Person.objects().order_by('age')
        assert qs_disk.count() == qs.count()
        for index in range(qs_disk.count()):
            assert qs_disk[index] == qs[index]
if __name__ == '__main__':
    unittest.main()