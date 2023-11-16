"""
Simple example with a single Document demonstrating how schema can be managed,
including upgrading with reindexing.

Key concepts:

    * setup() function to first initialize the schema (as index template) in
      elasticsearch. Can be called any time (recommended with every deploy of
      your app).

    * migrate() function to be called any time when the schema changes - it
      will create a new index (by incrementing the version) and update the alias.
      By default it will also (before flipping the alias) move the data from the
      previous index to the new one.

    * BlogPost._matches() class method is required for this code to work since
      otherwise BlogPost will not be used to deserialize the documents as those
      will have index set to the concrete index whereas the class refers to the
      alias.
"""
from datetime import datetime
from fnmatch import fnmatch
from elasticsearch_dsl import Date, Document, Keyword, Text, connections
ALIAS = 'test-blog'
PATTERN = ALIAS + '-*'

class BlogPost(Document):
    title = Text()
    published = Date()
    tags = Keyword(multi=True)
    content = Text()

    def is_published(self):
        if False:
            for i in range(10):
                print('nop')
        return self.published and datetime.now() > self.published

    @classmethod
    def _matches(cls, hit):
        if False:
            while True:
                i = 10
        return fnmatch(hit['_index'], PATTERN)

    class Index:
        name = ALIAS
        settings = {'number_of_shards': 1, 'number_of_replicas': 0}

def setup():
    if False:
        while True:
            i = 10
    '\n    Create the index template in elasticsearch specifying the mappings and any\n    settings to be used. This can be run at any time, ideally at every new code\n    deploy.\n    '
    index_template = BlogPost._index.as_template(ALIAS, PATTERN)
    index_template.save()
    if not BlogPost._index.exists():
        migrate(move_data=False)

def migrate(move_data=True, update_alias=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Upgrade function that creates a new index for the data. Optionally it also can\n    (and by default will) reindex previous copy of the data into the new index\n    (specify ``move_data=False`` to skip this step) and update the alias to\n    point to the latest index (set ``update_alias=False`` to skip).\n\n    Note that while this function is running the application can still perform\n    any and all searches without any loss of functionality. It should, however,\n    not perform any writes at this time as those might be lost.\n    '
    next_index = PATTERN.replace('*', datetime.now().strftime('%Y%m%d%H%M%S%f'))
    es = connections.get_connection()
    es.indices.create(index=next_index)
    if move_data:
        es.options(request_timeout=3600).reindex(body={'source': {'index': ALIAS}, 'dest': {'index': next_index}})
        es.indices.refresh(index=next_index)
    if update_alias:
        es.indices.update_aliases(body={'actions': [{'remove': {'alias': ALIAS, 'index': PATTERN}}, {'add': {'alias': ALIAS, 'index': next_index}}]})
if __name__ == '__main__':
    connections.create_connection()
    setup()
    bp = BlogPost(_id=0, title='Hello World!', tags=['testing', 'dummy'], content=open(__file__).read())
    bp.save(refresh=True)
    migrate()