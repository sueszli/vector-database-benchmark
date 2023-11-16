from haystack.query import SearchQuerySet
from blog.models import Article, Category

class BlogApi:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.searchqueryset = SearchQuerySet()
        self.searchqueryset.auto_query('')
        self.__max_takecount__ = 8

    def search_articles(self, query):
        if False:
            for i in range(10):
                print('nop')
        sqs = self.searchqueryset.auto_query(query)
        sqs = sqs.load_all()
        return sqs[:self.__max_takecount__]

    def get_category_lists(self):
        if False:
            while True:
                i = 10
        return Category.objects.all()

    def get_category_articles(self, categoryname):
        if False:
            return 10
        articles = Article.objects.filter(category__name=categoryname)
        if articles:
            return articles[:self.__max_takecount__]
        return None

    def get_recent_articles(self):
        if False:
            return 10
        return Article.objects.all()[:self.__max_takecount__]