from datetime import datetime
import pytest
from django_filters import FilterSet, rest_framework as filters
from graphene import ObjectType, Schema
from graphene.relay import Node
from graphene_django import DjangoObjectType
from graphene_django.filter.tests.filters import ArticleFilter
from graphene_django.tests.models import Article, Film, Person, Pet, Reporter
from graphene_django.utils import DJANGO_FILTER_INSTALLED
pytestmark = []
if DJANGO_FILTER_INSTALLED:
    from graphene_django.filter import DjangoFilterConnectionField
else:
    pytestmark.append(pytest.mark.skipif(True, reason='django_filters not installed or not compatible'))

@pytest.fixture
def query():
    if False:
        for i in range(10):
            print('nop')

    class PetNode(DjangoObjectType):

        class Meta:
            model = Pet
            interfaces = (Node,)
            fields = '__all__'
            filter_fields = {'id': ['exact', 'in'], 'name': ['exact', 'in'], 'age': ['exact', 'in', 'range']}

    class ReporterNode(DjangoObjectType):

        class Meta:
            model = Reporter
            interfaces = (Node,)
            fields = '__all__'
            filter_fields = {'reporter_type': ['exact', 'in']}

    class ArticleNode(DjangoObjectType):

        class Meta:
            model = Article
            interfaces = (Node,)
            fields = '__all__'
            filterset_class = ArticleFilter

    class FilmNode(DjangoObjectType):

        class Meta:
            model = Film
            interfaces = (Node,)
            fields = '__all__'
            filter_fields = {'genre': ['exact', 'in']}
            convert_choices_to_enum = False

    class PersonFilterSet(FilterSet):

        class Meta:
            model = Person
            fields = {'name': ['in']}
        names = filters.BaseInFilter(method='filter_names')

        def filter_names(self, qs, name, value):
            if False:
                i = 10
                return i + 15
            '\n            This custom filter take a string as input with comma separated values.\n            Note that the value here is already a list as it has been transformed by the BaseInFilter class.\n            '
            return qs.filter(name__in=value)

    class PersonNode(DjangoObjectType):

        class Meta:
            model = Person
            interfaces = (Node,)
            filterset_class = PersonFilterSet
            fields = '__all__'

    class Query(ObjectType):
        pets = DjangoFilterConnectionField(PetNode)
        people = DjangoFilterConnectionField(PersonNode)
        articles = DjangoFilterConnectionField(ArticleNode)
        films = DjangoFilterConnectionField(FilmNode)
        reporters = DjangoFilterConnectionField(ReporterNode)
    return Query

def test_string_in_filter(query):
    if False:
        while True:
            i = 10
    '\n    Test in filter on a string field.\n    '
    Pet.objects.create(name='Brutus', age=12)
    Pet.objects.create(name='Mimi', age=3)
    Pet.objects.create(name='Jojo, the rabbit', age=3)
    schema = Schema(query=query)
    query = '\n    query {\n        pets (name_In: ["Brutus", "Jojo, the rabbit"]) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['pets']['edges'] == [{'node': {'name': 'Brutus'}}, {'node': {'name': 'Jojo, the rabbit'}}]

def test_string_in_filter_with_otjer_filter(query):
    if False:
        while True:
            i = 10
    '\n    Test in filter on a string field which has also a custom filter doing a similar operation.\n    '
    Person.objects.create(name='John')
    Person.objects.create(name='Michael')
    Person.objects.create(name='Angela')
    schema = Schema(query=query)
    query = '\n    query {\n        people (name_In: ["John", "Michael"]) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['people']['edges'] == [{'node': {'name': 'John'}}, {'node': {'name': 'Michael'}}]

def test_string_in_filter_with_declared_filter(query):
    if False:
        print('Hello World!')
    '\n    Test in filter on a string field with a custom filterset class.\n    '
    Person.objects.create(name='John')
    Person.objects.create(name='Michael')
    Person.objects.create(name='Angela')
    schema = Schema(query=query)
    query = '\n    query {\n        people (names: "John,Michael") {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['people']['edges'] == [{'node': {'name': 'John'}}, {'node': {'name': 'Michael'}}]

def test_int_in_filter(query):
    if False:
        print('Hello World!')
    '\n    Test in filter on an integer field.\n    '
    Pet.objects.create(name='Brutus', age=12)
    Pet.objects.create(name='Mimi', age=3)
    Pet.objects.create(name='Jojo, the rabbit', age=3)
    schema = Schema(query=query)
    query = '\n    query {\n        pets (age_In: [3]) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['pets']['edges'] == [{'node': {'name': 'Mimi'}}, {'node': {'name': 'Jojo, the rabbit'}}]
    query = '\n    query {\n        pets (age_In: [3, 12]) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['pets']['edges'] == [{'node': {'name': 'Brutus'}}, {'node': {'name': 'Mimi'}}, {'node': {'name': 'Jojo, the rabbit'}}]

def test_in_filter_with_empty_list(query):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check that using a in filter with an empty list provided as input returns no objects.\n    '
    Pet.objects.create(name='Brutus', age=12)
    Pet.objects.create(name='Mimi', age=8)
    Pet.objects.create(name='Picotin', age=5)
    schema = Schema(query=query)
    query = '\n    query {\n        pets (name_In: []) {\n            edges {\n                node {\n                    name\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert len(result.data['pets']['edges']) == 0

def test_choice_in_filter_without_enum(query):
    if False:
        return 10
    '\n    Test in filter o an choice field not using an enum (Film.genre).\n    '
    john_doe = Reporter.objects.create(first_name='John', last_name='Doe', email='john@doe.com')
    jean_bon = Reporter.objects.create(first_name='Jean', last_name='Bon', email='jean@bon.com')
    documentary_film = Film.objects.create(genre='do')
    documentary_film.reporters.add(john_doe)
    action_film = Film.objects.create(genre='ac')
    action_film.reporters.add(john_doe)
    other_film = Film.objects.create(genre='ot')
    other_film.reporters.add(john_doe)
    other_film.reporters.add(jean_bon)
    schema = Schema(query=query)
    query = '\n    query {\n        films (genre_In: ["do", "ac"]) {\n            edges {\n                node {\n                    genre\n                    reporters {\n                        edges {\n                            node {\n                                lastName\n                            }\n                        }\n                    }\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['films']['edges'] == [{'node': {'genre': 'do', 'reporters': {'edges': [{'node': {'lastName': 'Doe'}}]}}}, {'node': {'genre': 'ac', 'reporters': {'edges': [{'node': {'lastName': 'Doe'}}]}}}]

def test_fk_id_in_filter(query):
    if False:
        print('Hello World!')
    '\n    Test in filter on an foreign key relationship.\n    '
    john_doe = Reporter.objects.create(first_name='John', last_name='Doe', email='john@doe.com')
    jean_bon = Reporter.objects.create(first_name='Jean', last_name='Bon', email='jean@bon.com')
    sara_croche = Reporter.objects.create(first_name='Sara', last_name='Croche', email='sara@croche.com')
    Article.objects.create(headline='A', pub_date=datetime.now(), pub_date_time=datetime.now(), reporter=john_doe, editor=john_doe)
    Article.objects.create(headline='B', pub_date=datetime.now(), pub_date_time=datetime.now(), reporter=jean_bon, editor=jean_bon)
    Article.objects.create(headline='C', pub_date=datetime.now(), pub_date_time=datetime.now(), reporter=sara_croche, editor=sara_croche)
    schema = Schema(query=query)
    query = f'\n    query {{\n        articles (reporter_In: [{john_doe.id}, {jean_bon.id}]) {{\n            edges {{\n                node {{\n                    headline\n                    reporter {{\n                        lastName\n                    }}\n                }}\n            }}\n        }}\n    }}\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['articles']['edges'] == [{'node': {'headline': 'A', 'reporter': {'lastName': 'Doe'}}}, {'node': {'headline': 'B', 'reporter': {'lastName': 'Bon'}}}]

def test_enum_in_filter(query):
    if False:
        print('Hello World!')
    '\n    Test in filter on a choice field using an enum (Reporter.reporter_type).\n    '
    Reporter.objects.create(first_name='John', last_name='Doe', email='john@doe.com', reporter_type=1)
    Reporter.objects.create(first_name='Jean', last_name='Bon', email='jean@bon.com', reporter_type=2)
    Reporter.objects.create(first_name='Jane', last_name='Doe', email='jane@doe.com', reporter_type=2)
    Reporter.objects.create(first_name='Jack', last_name='Black', email='jack@black.com', reporter_type=None)
    schema = Schema(query=query)
    query = '\n    query {\n        reporters (reporterType_In: [A_1]) {\n            edges {\n                node {\n                    email\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['reporters']['edges'] == [{'node': {'email': 'john@doe.com'}}]
    query = '\n    query {\n        reporters (reporterType_In: [A_2]) {\n            edges {\n                node {\n                    email\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['reporters']['edges'] == [{'node': {'email': 'jean@bon.com'}}, {'node': {'email': 'jane@doe.com'}}]
    query = '\n    query {\n        reporters (reporterType_In: [A_2, A_1]) {\n            edges {\n                node {\n                    email\n                }\n            }\n        }\n    }\n    '
    result = schema.execute(query)
    assert not result.errors
    assert result.data['reporters']['edges'] == [{'node': {'email': 'john@doe.com'}}, {'node': {'email': 'jean@bon.com'}}, {'node': {'email': 'jane@doe.com'}}]