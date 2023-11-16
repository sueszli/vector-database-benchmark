import pytest
from graphql_relay import to_global_id
import graphene
from graphene.relay import Node
from ..types import DjangoObjectType
from .models import Article, Film, FilmDetails, Reporter

class TestShouldCallGetQuerySetOnForeignKey:
    """
    Check that the get_queryset method is called in both forward and reversed direction
    of a foreignkey on types.
    (see issue #1111)

    NOTE: For now, we do not expect this get_queryset method to be called for nested
    objects, as the original attempt to do so prevented SQL query-optimization with
    `select_related`/`prefetch_related` and caused N+1 queries. See discussions here
    https://github.com/graphql-python/graphene-django/pull/1315/files#r1015659857
    and here https://github.com/graphql-python/graphene-django/pull/1401.
    """

    @pytest.fixture(autouse=True)
    def setup_schema(self):
        if False:
            while True:
                i = 10

        class ReporterType(DjangoObjectType):

            class Meta:
                model = Reporter

            @classmethod
            def get_queryset(cls, queryset, info):
                if False:
                    return 10
                if info.context and info.context.get('admin'):
                    return queryset
                raise Exception('Not authorized to access reporters.')

        class ArticleType(DjangoObjectType):

            class Meta:
                model = Article

            @classmethod
            def get_queryset(cls, queryset, info):
                if False:
                    while True:
                        i = 10
                return queryset.exclude(headline__startswith='Draft')

        class Query(graphene.ObjectType):
            reporter = graphene.Field(ReporterType, id=graphene.ID(required=True))
            article = graphene.Field(ArticleType, id=graphene.ID(required=True))

            def resolve_reporter(self, info, id):
                if False:
                    return 10
                return ReporterType.get_queryset(Reporter.objects, info).filter(id=id).last()

            def resolve_article(self, info, id):
                if False:
                    return 10
                return ArticleType.get_queryset(Article.objects, info).filter(id=id).last()
        self.schema = graphene.Schema(query=Query)
        self.reporter = Reporter.objects.create(first_name='Jane', last_name='Doe')
        self.articles = [Article.objects.create(headline='A fantastic article', reporter=self.reporter, editor=self.reporter), Article.objects.create(headline='Draft: My next best seller', reporter=self.reporter, editor=self.reporter)]

    def test_get_queryset_called_on_field(self):
        if False:
            return 10
        query = '\n            query getArticle($id: ID!) {\n                article(id: $id) {\n                    headline\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': self.articles[0].id})
        assert not result.errors
        assert result.data['article'] == {'headline': 'A fantastic article'}
        result = self.schema.execute(query, variables={'id': self.articles[1].id})
        assert not result.errors
        assert result.data['article'] is None
        query = '\n            query getReporter($id: ID!) {\n                reporter(id: $id) {\n                    firstName\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': self.reporter.id})
        assert len(result.errors) == 1
        assert result.errors[0].message == 'Not authorized to access reporters.'
        query = '\n            query getReporter($id: ID!) {\n                reporter(id: $id) {\n                    firstName\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': self.reporter.id}, context_value={'admin': True})
        assert not result.errors
        assert result.data == {'reporter': {'firstName': 'Jane'}}

    def test_get_queryset_called_on_foreignkey(self):
        if False:
            for i in range(10):
                print('nop')
        query = '\n            query getArticle($id: ID!) {\n                article(id: $id) {\n                    headline\n                    reporter {\n                        firstName\n                    }\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': self.articles[0].id})
        assert len(result.errors) == 1
        assert result.errors[0].message == 'Not authorized to access reporters.'
        query = '\n            query getArticle($id: ID!) {\n                article(id: $id) {\n                    headline\n                    reporter {\n                        firstName\n                    }\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': self.articles[0].id}, context_value={'admin': True})
        assert not result.errors
        assert result.data['article'] == {'headline': 'A fantastic article', 'reporter': {'firstName': 'Jane'}}
        query = '\n            query getReporter($id: ID!) {\n                reporter(id: $id) {\n                    firstName\n                    articles {\n                        headline\n                    }\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': self.reporter.id}, context_value={'admin': True})
        assert not result.errors
        assert result.data['reporter'] == {'firstName': 'Jane', 'articles': [{'headline': 'A fantastic article'}]}

class TestShouldCallGetQuerySetOnForeignKeyNode:
    """
    Check that the get_queryset method is called in both forward and reversed direction
    of a foreignkey on types using a node interface.
    (see issue #1111)
    """

    @pytest.fixture(autouse=True)
    def setup_schema(self):
        if False:
            print('Hello World!')

        class ReporterType(DjangoObjectType):

            class Meta:
                model = Reporter
                interfaces = (Node,)

            @classmethod
            def get_queryset(cls, queryset, info):
                if False:
                    i = 10
                    return i + 15
                if info.context and info.context.get('admin'):
                    return queryset
                raise Exception('Not authorized to access reporters.')

        class ArticleType(DjangoObjectType):

            class Meta:
                model = Article
                interfaces = (Node,)

            @classmethod
            def get_queryset(cls, queryset, info):
                if False:
                    for i in range(10):
                        print('nop')
                return queryset.exclude(headline__startswith='Draft')

        class Query(graphene.ObjectType):
            reporter = Node.Field(ReporterType)
            article = Node.Field(ArticleType)
        self.schema = graphene.Schema(query=Query)
        self.reporter = Reporter.objects.create(first_name='Jane', last_name='Doe')
        self.articles = [Article.objects.create(headline='A fantastic article', reporter=self.reporter, editor=self.reporter), Article.objects.create(headline='Draft: My next best seller', reporter=self.reporter, editor=self.reporter)]

    def test_get_queryset_called_on_node(self):
        if False:
            i = 10
            return i + 15
        query = '\n            query getArticle($id: ID!) {\n                article(id: $id) {\n                    headline\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': to_global_id('ArticleType', self.articles[0].id)})
        assert not result.errors
        assert result.data['article'] == {'headline': 'A fantastic article'}
        result = self.schema.execute(query, variables={'id': to_global_id('ArticleType', self.articles[1].id)})
        assert not result.errors
        assert result.data['article'] is None
        query = '\n            query getReporter($id: ID!) {\n                reporter(id: $id) {\n                    firstName\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': to_global_id('ReporterType', self.reporter.id)})
        assert len(result.errors) == 1
        assert result.errors[0].message == 'Not authorized to access reporters.'
        query = '\n            query getReporter($id: ID!) {\n                reporter(id: $id) {\n                    firstName\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': to_global_id('ReporterType', self.reporter.id)}, context_value={'admin': True})
        assert not result.errors
        assert result.data == {'reporter': {'firstName': 'Jane'}}

    def test_get_queryset_called_on_foreignkey(self):
        if False:
            print('Hello World!')
        query = '\n            query getArticle($id: ID!) {\n                article(id: $id) {\n                    headline\n                    reporter {\n                        firstName\n                    }\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': to_global_id('ArticleType', self.articles[0].id)})
        assert len(result.errors) == 1
        assert result.errors[0].message == 'Not authorized to access reporters.'
        query = '\n            query getArticle($id: ID!) {\n                article(id: $id) {\n                    headline\n                    reporter {\n                        firstName\n                    }\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': to_global_id('ArticleType', self.articles[0].id)}, context_value={'admin': True})
        assert not result.errors
        assert result.data['article'] == {'headline': 'A fantastic article', 'reporter': {'firstName': 'Jane'}}
        query = '\n            query getReporter($id: ID!) {\n                reporter(id: $id) {\n                    firstName\n                    articles {\n                        edges {\n                            node {\n                                headline\n                            }\n                        }\n                    }\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': to_global_id('ReporterType', self.reporter.id)}, context_value={'admin': True})
        assert not result.errors
        assert result.data['reporter'] == {'firstName': 'Jane', 'articles': {'edges': [{'node': {'headline': 'A fantastic article'}}]}}

class TestShouldCallGetQuerySetOnOneToOne:

    @pytest.fixture(autouse=True)
    def setup_schema(self):
        if False:
            i = 10
            return i + 15

        class FilmDetailsType(DjangoObjectType):

            class Meta:
                model = FilmDetails

            @classmethod
            def get_queryset(cls, queryset, info):
                if False:
                    i = 10
                    return i + 15
                if info.context and info.context.get('permission_get_film_details'):
                    return queryset
                raise Exception('Not authorized to access film details.')

        class FilmType(DjangoObjectType):

            class Meta:
                model = Film

            @classmethod
            def get_queryset(cls, queryset, info):
                if False:
                    for i in range(10):
                        print('nop')
                if info.context and info.context.get('permission_get_film'):
                    return queryset
                raise Exception('Not authorized to access film.')

        class Query(graphene.ObjectType):
            film_details = graphene.Field(FilmDetailsType, id=graphene.ID(required=True))
            film = graphene.Field(FilmType, id=graphene.ID(required=True))

            def resolve_film_details(self, info, id):
                if False:
                    return 10
                return FilmDetailsType.get_queryset(FilmDetails.objects, info).filter(id=id).last()

            def resolve_film(self, info, id):
                if False:
                    for i in range(10):
                        print('nop')
                return FilmType.get_queryset(Film.objects, info).filter(id=id).last()
        self.schema = graphene.Schema(query=Query)
        self.films = [Film.objects.create(genre='do'), Film.objects.create(genre='ac')]
        self.film_details = [FilmDetails.objects.create(film=self.films[0]), FilmDetails.objects.create(film=self.films[1])]

    def test_get_queryset_called_on_field(self):
        if False:
            return 10
        query = '\n            query getFilm($id: ID!) {\n                film(id: $id) {\n                    genre\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': self.films[0].id}, context_value={'permission_get_film': True})
        assert not result.errors
        assert result.data['film'] == {'genre': 'DO'}
        result = self.schema.execute(query, variables={'id': self.films[1].id}, context_value={'permission_get_film': False})
        assert len(result.errors) == 1
        assert result.errors[0].message == 'Not authorized to access film.'
        query = '\n            query getFilmDetails($id: ID!) {\n                filmDetails(id: $id) {\n                    location\n                }\n            }\n        '
        result = self.schema.execute(query, variables={'id': self.film_details[0].id}, context_value={'permission_get_film_details': True})
        assert not result.errors
        assert result.data == {'filmDetails': {'location': ''}}
        result = self.schema.execute(query, variables={'id': self.film_details[0].id}, context_value={'permission_get_film_details': False})
        assert len(result.errors) == 1
        assert result.errors[0].message == 'Not authorized to access film details.'

    def test_get_queryset_called_on_foreignkey(self, django_assert_num_queries):
        if False:
            print('Hello World!')
        query = '\n            query getFilm($id: ID!) {\n                film(id: $id) {\n                    genre\n                    details {\n                        location\n                    }\n                }\n            }\n        '
        with django_assert_num_queries(2):
            result = self.schema.execute(query, variables={'id': self.films[0].id}, context_value={'permission_get_film': True, 'permission_get_film_details': True})
        assert not result.errors
        assert result.data['film'] == {'genre': 'DO', 'details': {'location': ''}}
        with django_assert_num_queries(1):
            result = self.schema.execute(query, variables={'id': self.films[0].id}, context_value={'permission_get_film': True, 'permission_get_film_details': False})
        assert len(result.errors) == 1
        assert result.errors[0].message == 'Not authorized to access film details.'
        query = '\n            query getFilmDetails($id: ID!) {\n                filmDetails(id: $id) {\n                    location\n                    film {\n                        genre\n                    }\n                }\n            }\n        '
        with django_assert_num_queries(2):
            result = self.schema.execute(query, variables={'id': self.film_details[0].id}, context_value={'permission_get_film': True, 'permission_get_film_details': True})
        assert not result.errors
        assert result.data['filmDetails'] == {'location': '', 'film': {'genre': 'DO'}}
        with django_assert_num_queries(1):
            result = self.schema.execute(query, variables={'id': self.film_details[1].id}, context_value={'permission_get_film': False, 'permission_get_film_details': True})
        assert len(result.errors) == 1
        assert result.errors[0].message == 'Not authorized to access film.'