from haystack import indexes
from blog.models import Article

class ArticleIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, use_template=True)

    def get_model(self):
        if False:
            return 10
        return Article

    def index_queryset(self, using=None):
        if False:
            while True:
                i = 10
        return self.get_model().objects.filter(status='p')