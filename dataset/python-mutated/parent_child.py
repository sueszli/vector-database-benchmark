"""
Complex data model example modeling stackoverflow-like data.

It is used to showcase several key features of elasticsearch-dsl:

    * Object and Nested fields: see User and Comment classes and fields they
      are used in

        * method add_comment is used to add comments

    * Parent/Child relationship

        * See the Join field on Post creating the relationship between Question
          and Answer

        * Meta.matches allows the hits from same index to be wrapped in proper
          classes

        * to see how child objects are created see Question.add_answer

        * Question.search_answers shows how to query for children of a
          particular parent

"""
from datetime import datetime
from elasticsearch_dsl import Boolean, Date, Document, InnerDoc, Join, Keyword, Long, Nested, Object, Text, connections

class User(InnerDoc):
    """
    Class used to represent a denormalized user stored on other objects.
    """
    id = Long(required=True)
    signed_up = Date()
    username = Text(fields={'keyword': Keyword()}, required=True)
    email = Text(fields={'keyword': Keyword()})
    location = Text(fields={'keyword': Keyword()})

class Comment(InnerDoc):
    """
    Class wrapper for nested comment objects.
    """
    author = Object(User, required=True)
    created = Date(required=True)
    content = Text(required=True)

class Post(Document):
    """
    Base class for Question and Answer containing the common fields.
    """
    author = Object(User, required=True)
    created = Date(required=True)
    body = Text(required=True)
    comments = Nested(Comment)
    question_answer = Join(relations={'question': 'answer'})

    @classmethod
    def _matches(cls, hit):
        if False:
            i = 10
            return i + 15
        return False

    class Index:
        name = 'test-qa-site'
        settings = {'number_of_shards': 1, 'number_of_replicas': 0}

    def add_comment(self, user, content, created=None, commit=True):
        if False:
            while True:
                i = 10
        c = Comment(author=user, content=content, created=created or datetime.now())
        self.comments.append(c)
        if commit:
            self.save()
        return c

    def save(self, **kwargs):
        if False:
            return 10
        if self.created is None:
            self.created = datetime.now()
        return super().save(**kwargs)

class Question(Post):
    tags = Keyword(multi=True)
    title = Text(fields={'keyword': Keyword()})

    @classmethod
    def _matches(cls, hit):
        if False:
            while True:
                i = 10
        'Use Question class for parent documents'
        return hit['_source']['question_answer'] == 'question'

    @classmethod
    def search(cls, **kwargs):
        if False:
            while True:
                i = 10
        return cls._index.search(**kwargs).filter('term', question_answer='question')

    def add_answer(self, user, body, created=None, accepted=False, commit=True):
        if False:
            i = 10
            return i + 15
        answer = Answer(_routing=self.meta.id, _index=self.meta.index, question_answer={'name': 'answer', 'parent': self.meta.id}, author=user, created=created, body=body, accepted=accepted)
        if commit:
            answer.save()
        return answer

    def search_answers(self):
        if False:
            print('Hello World!')
        s = Answer.search()
        s = s.filter('parent_id', type='answer', id=self.meta.id)
        s = s.params(routing=self.meta.id)
        return s

    def get_answers(self):
        if False:
            return 10
        '\n        Get answers either from inner_hits already present or by searching\n        elasticsearch.\n        '
        if 'inner_hits' in self.meta and 'answer' in self.meta.inner_hits:
            return self.meta.inner_hits.answer.hits
        return list(self.search_answers())

    def save(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.question_answer = 'question'
        return super().save(**kwargs)

class Answer(Post):
    is_accepted = Boolean()

    @classmethod
    def _matches(cls, hit):
        if False:
            for i in range(10):
                print('nop')
        "Use Answer class for child documents with child name 'answer'"
        return isinstance(hit['_source']['question_answer'], dict) and hit['_source']['question_answer'].get('name') == 'answer'

    @classmethod
    def search(cls, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return cls._index.search(**kwargs).exclude('term', question_answer='question')

    @property
    def question(self):
        if False:
            while True:
                i = 10
        if 'question' not in self.meta:
            self.meta.question = Question.get(id=self.question_answer.parent, index=self.meta.index)
        return self.meta.question

    def save(self, **kwargs):
        if False:
            print('Hello World!')
        self.meta.routing = self.question_answer.parent
        return super().save(**kwargs)

def setup():
    if False:
        i = 10
        return i + 15
    'Create an IndexTemplate and save it into elasticsearch.'
    index_template = Post._index.as_template('base')
    index_template.save()
if __name__ == '__main__':
    connections.create_connection()
    setup()
    nick = User(id=47, signed_up=datetime(2017, 4, 3), username='fxdgear', email='nick.lang@elastic.co', location='Colorado')
    honza = User(id=42, signed_up=datetime(2013, 4, 3), username='honzakral', email='honza@elastic.co', location='Prague')
    question = Question(_id=1, author=nick, tags=['elasticsearch', 'python'], title='How do I use elasticsearch from Python?', body='\n        I want to use elasticsearch, how do I do it from Python?\n        ')
    question.save()
    answer = question.add_answer(honza, 'Just use `elasticsearch-py`!')