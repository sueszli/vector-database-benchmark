from django.utils.encoding import force_str
from elasticsearch_dsl import Q
from haystack.backends import BaseEngine, BaseSearchBackend, BaseSearchQuery, log_query
from haystack.forms import ModelSearchForm
from haystack.models import SearchResult
from haystack.utils import log as logging
from blog.documents import ArticleDocument, ArticleDocumentManager
from blog.models import Article
logger = logging.getLogger(__name__)

class ElasticSearchBackend(BaseSearchBackend):

    def __init__(self, connection_alias, **connection_options):
        if False:
            print('Hello World!')
        super(ElasticSearchBackend, self).__init__(connection_alias, **connection_options)
        self.manager = ArticleDocumentManager()
        self.include_spelling = True

    def _get_models(self, iterable):
        if False:
            i = 10
            return i + 15
        models = iterable if iterable and iterable[0] else Article.objects.all()
        docs = self.manager.convert_to_doc(models)
        return docs

    def _create(self, models):
        if False:
            return 10
        self.manager.create_index()
        docs = self._get_models(models)
        self.manager.rebuild(docs)

    def _delete(self, models):
        if False:
            return 10
        for m in models:
            m.delete()
        return True

    def _rebuild(self, models):
        if False:
            while True:
                i = 10
        models = models if models else Article.objects.all()
        docs = self.manager.convert_to_doc(models)
        self.manager.update_docs(docs)

    def update(self, index, iterable, commit=True):
        if False:
            while True:
                i = 10
        models = self._get_models(iterable)
        self.manager.update_docs(models)

    def remove(self, obj_or_string):
        if False:
            print('Hello World!')
        models = self._get_models([obj_or_string])
        self._delete(models)

    def clear(self, models=None, commit=True):
        if False:
            i = 10
            return i + 15
        self.remove(None)

    @staticmethod
    def get_suggestion(query: str) -> str:
        if False:
            return 10
        '获取推荐词, 如果没有找到添加原搜索词'
        search = ArticleDocument.search().query('match', body=query).suggest('suggest_search', query, term={'field': 'body'}).execute()
        keywords = []
        for suggest in search.suggest.suggest_search:
            if suggest['options']:
                keywords.append(suggest['options'][0]['text'])
            else:
                keywords.append(suggest['text'])
        return ' '.join(keywords)

    @log_query
    def search(self, query_string, **kwargs):
        if False:
            return 10
        logger.info('search query_string:' + query_string)
        start_offset = kwargs.get('start_offset')
        end_offset = kwargs.get('end_offset')
        if getattr(self, 'is_suggest', None):
            suggestion = self.get_suggestion(query_string)
        else:
            suggestion = query_string
        q = Q('bool', should=[Q('match', body=suggestion), Q('match', title=suggestion)], minimum_should_match='70%')
        search = ArticleDocument.search().query('bool', filter=[q]).filter('term', status='p').filter('term', type='a').source(False)[start_offset:end_offset]
        results = search.execute()
        hits = results['hits'].total
        raw_results = []
        for raw_result in results['hits']['hits']:
            app_label = 'blog'
            model_name = 'Article'
            additional_fields = {}
            result_class = SearchResult
            result = result_class(app_label, model_name, raw_result['_id'], raw_result['_score'], **additional_fields)
            raw_results.append(result)
        facets = {}
        spelling_suggestion = None if query_string == suggestion else suggestion
        return {'results': raw_results, 'hits': hits, 'facets': facets, 'spelling_suggestion': spelling_suggestion}

class ElasticSearchQuery(BaseSearchQuery):

    def _convert_datetime(self, date):
        if False:
            print('Hello World!')
        if hasattr(date, 'hour'):
            return force_str(date.strftime('%Y%m%d%H%M%S'))
        else:
            return force_str(date.strftime('%Y%m%d000000'))

    def clean(self, query_fragment):
        if False:
            print('Hello World!')
        '\n        Provides a mechanism for sanitizing user input before presenting the\n        value to the backend.\n\n        Whoosh 1.X differs here in that you can no longer use a backslash\n        to escape reserved characters. Instead, the whole word should be\n        quoted.\n        '
        words = query_fragment.split()
        cleaned_words = []
        for word in words:
            if word in self.backend.RESERVED_WORDS:
                word = word.replace(word, word.lower())
            for char in self.backend.RESERVED_CHARACTERS:
                if char in word:
                    word = "'%s'" % word
                    break
            cleaned_words.append(word)
        return ' '.join(cleaned_words)

    def build_query_fragment(self, field, filter_type, value):
        if False:
            print('Hello World!')
        return value.query_string

    def get_count(self):
        if False:
            i = 10
            return i + 15
        results = self.get_results()
        return len(results) if results else 0

    def get_spelling_suggestion(self, preferred_query=None):
        if False:
            while True:
                i = 10
        return self._spelling_suggestion

    def build_params(self, spelling_query=None):
        if False:
            return 10
        kwargs = super(ElasticSearchQuery, self).build_params(spelling_query=spelling_query)
        return kwargs

class ElasticSearchModelSearchForm(ModelSearchForm):

    def search(self):
        if False:
            while True:
                i = 10
        self.searchqueryset.query.backend.is_suggest = self.data.get('is_suggest') != 'no'
        sqs = super().search()
        return sqs

class ElasticSearchEngine(BaseEngine):
    backend = ElasticSearchBackend
    query = ElasticSearchQuery