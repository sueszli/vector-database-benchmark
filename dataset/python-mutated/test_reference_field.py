import pytest
from bson import SON, DBRef
from mongoengine import *
from tests.utils import MongoDBTestCase

class TestReferenceField(MongoDBTestCase):

    def test_reference_validation(self):
        if False:
            i = 10
            return i + 15
        'Ensure that invalid document objects cannot be assigned to\n        reference fields.\n        '

        class User(Document):
            name = StringField()

        class BlogPost(Document):
            content = StringField()
            author = ReferenceField(User)
        User.drop_collection()
        BlogPost.drop_collection()
        with pytest.raises(ValidationError):
            ReferenceField(EmbeddedDocument)
        user = User(name='Test User')
        post1 = BlogPost(content='Chips and gravy taste good.')
        post1.author = user
        with pytest.raises(ValidationError):
            post1.save()
        post2 = BlogPost(content='Chips and chilli taste good.')
        post1.author = post2
        with pytest.raises(ValidationError):
            post1.validate()
        user_object_id = user.pk
        post3 = BlogPost(content='Chips and curry sauce taste good.')
        post3.author = user_object_id
        post3.save()
        user.save()
        post1.author = user
        post1.save()
        post2.save()
        post1.author = post2
        with pytest.raises(ValidationError):
            post1.validate()

    def test_dbref_reference_fields(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure storing references as bson.dbref.DBRef works.'

        class Person(Document):
            name = StringField()
            parent = ReferenceField('self', dbref=True)
        Person.drop_collection()
        p1 = Person(name='John').save()
        Person(name='Ross', parent=p1).save()
        assert Person._get_collection().find_one({'name': 'Ross'})['parent'] == DBRef('person', p1.pk)
        p = Person.objects.get(name='Ross')
        assert p.parent == p1

    def test_dbref_to_mongo(self):
        if False:
            while True:
                i = 10
        'Make sure that calling to_mongo on a ReferenceField which\n        has dbref=False, but actually actually contains a DBRef returns\n        an ID of that DBRef.\n        '

        class Person(Document):
            name = StringField()
            parent = ReferenceField('self', dbref=False)
        p = Person(name='Steve', parent=DBRef('person', 'abcdefghijklmnop'))
        assert p.to_mongo() == SON([('name', 'Steve'), ('parent', 'abcdefghijklmnop')])

    def test_objectid_reference_fields(self):
        if False:
            for i in range(10):
                print('nop')

        class Person(Document):
            name = StringField()
            parent = ReferenceField('self', dbref=False)
        Person.drop_collection()
        p1 = Person(name='John').save()
        Person(name='Ross', parent=p1).save()
        col = Person._get_collection()
        data = col.find_one({'name': 'Ross'})
        assert data['parent'] == p1.pk
        p = Person.objects.get(name='Ross')
        assert p.parent == p1

    def test_undefined_reference(self):
        if False:
            i = 10
            return i + 15
        'Ensure that ReferenceFields may reference undefined Documents.'

        class Product(Document):
            name = StringField()
            company = ReferenceField('Company')

        class Company(Document):
            name = StringField()
        Product.drop_collection()
        Company.drop_collection()
        ten_gen = Company(name='10gen')
        ten_gen.save()
        mongodb = Product(name='MongoDB', company=ten_gen)
        mongodb.save()
        me = Product(name='MongoEngine')
        me.save()
        obj = Product.objects(company=ten_gen).first()
        assert obj == mongodb
        assert obj.company == ten_gen
        obj = Product.objects(company=None).first()
        assert obj == me
        obj = Product.objects.get(company=None)
        assert obj == me

    def test_reference_query_conversion(self):
        if False:
            while True:
                i = 10
        'Ensure that ReferenceFields can be queried using objects and values\n        of the type of the primary key of the referenced object.\n        '

        class Member(Document):
            user_num = IntField(primary_key=True)

        class BlogPost(Document):
            title = StringField()
            author = ReferenceField(Member, dbref=False)
        Member.drop_collection()
        BlogPost.drop_collection()
        m1 = Member(user_num=1)
        m1.save()
        m2 = Member(user_num=2)
        m2.save()
        post1 = BlogPost(title='post 1', author=m1)
        post1.save()
        post2 = BlogPost(title='post 2', author=m2)
        post2.save()
        post = BlogPost.objects(author=m1).first()
        assert post.id == post1.id
        post = BlogPost.objects(author=m2).first()
        assert post.id == post2.id

    def test_reference_query_conversion_dbref(self):
        if False:
            print('Hello World!')
        'Ensure that ReferenceFields can be queried using objects and values\n        of the type of the primary key of the referenced object.\n        '

        class Member(Document):
            user_num = IntField(primary_key=True)

        class BlogPost(Document):
            title = StringField()
            author = ReferenceField(Member, dbref=True)
        Member.drop_collection()
        BlogPost.drop_collection()
        m1 = Member(user_num=1)
        m1.save()
        m2 = Member(user_num=2)
        m2.save()
        post1 = BlogPost(title='post 1', author=m1)
        post1.save()
        post2 = BlogPost(title='post 2', author=m2)
        post2.save()
        post = BlogPost.objects(author=m1).first()
        assert post.id == post1.id
        post = BlogPost.objects(author=m2).first()
        assert post.id == post2.id