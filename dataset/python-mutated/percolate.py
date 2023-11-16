from elasticsearch_dsl import Document, Keyword, Percolator, Q, Search, Text, connections

class BlogPost(Document):
    """
    Blog posts that will be automatically tagged based on percolation queries.
    """
    content = Text()
    tags = Keyword(multi=True)

    class Index:
        name = 'test-blogpost'

    def add_tags(self):
        if False:
            for i in range(10):
                print('nop')
        s = Search(index='test-percolator')
        s = s.query('percolate', field='query', index=self._get_index(), document=self.to_dict())
        for percolator in s:
            self.tags.extend(percolator.tags)
        self.tags = list(set(self.tags))

    def save(self, **kwargs):
        if False:
            return 10
        self.add_tags()
        return super().save(**kwargs)

class PercolatorDoc(Document):
    """
    Document class used for storing the percolation queries.
    """
    content = Text()
    query = Percolator()
    tags = Keyword(multi=True)

    class Index:
        name = 'test-percolator'
        settings = {'number_of_shards': 1, 'number_of_replicas': 0}

def setup():
    if False:
        while True:
            i = 10
    if not PercolatorDoc._index.exists():
        PercolatorDoc.init()
    PercolatorDoc(_id='python', tags=['programming', 'development', 'python'], query=Q('match', content='python')).save(refresh=True)
if __name__ == '__main__':
    connections.create_connection()
    setup()