"""
Common file database module tests
"""
import contextlib
import io
import os
import tempfile
import unittest
from txtai.embeddings import Embeddings, IndexNotFoundError
from txtai.database import Embedded, RDBMS, SQLError

class Common:
    """
    Wraps common file database tests to prevent unit test discovery for this class.
    """

    class TestRDBMS(unittest.TestCase):
        """
        Embeddings with content stored in a file database tests.
        """

        @classmethod
        def setUpClass(cls):
            if False:
                print('Hello World!')
            '\n            Initialize test data.\n            '
            cls.data = ['US tops 5 million confirmed virus cases', "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", 'Beijing mobilises invasion craft along coast as Taiwan tensions escalate', 'The National Park Service warns against sacrificing slower friends in a bear attack', 'Maine man wins $1M from $25 lottery ticket', 'Make huge profits without work, earn up to $100,000 a day']
            cls.backend = None
            cls.embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': cls.backend})

        @classmethod
        def tearDownClass(cls):
            if False:
                print('Hello World!')
            '\n            Cleanup data.\n            '
            if cls.embeddings:
                cls.embeddings.close()

        def testArchive(self):
            if False:
                print('Hello World!')
            '\n            Test embeddings index archiving\n            '
            for extension in ['tar.bz2', 'tar.gz', 'tar.xz', 'zip']:
                self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
                index = os.path.join(tempfile.gettempdir(), f'embeddings.{self.category()}.{extension}')
                self.embeddings.save(index)
                self.embeddings.load(index)
                result = self.embeddings.search('feel good story', 1)[0]
                self.assertEqual(result['text'], self.data[4])
                self.embeddings.upsert([(0, 'Looking out into the dreadful abyss', None)])
                self.assertEqual(self.embeddings.count(), len(self.data))

        def testAutoId(self):
            if False:
                while True:
                    i = 10
            '\n            Test auto id generation\n            '
            embeddings = Embeddings(path='sentence-transformers/nli-mpnet-base-v2', content=self.backend)
            embeddings.index(self.data)
            result = embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])
            embeddings.config['autoid'] = 'uuid4'
            embeddings.index(self.data)
            result = embeddings.search(self.data[4], 1)[0]
            self.assertEqual(len(result['id']), 36)

        def testColumns(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test custom text/object columns\n            '
            embeddings = Embeddings({'keyword': True, 'content': self.backend, 'columns': {'text': 'value'}})
            data = [{'value': x} for x in self.data]
            embeddings.index([(uid, text, None) for (uid, text) in enumerate(data)])
            result = embeddings.search('lottery', 1)[0]
            self.assertEqual(result['text'], self.data[4])

        def testClose(self):
            if False:
                i = 10
                return i + 15
            '\n            Test embeddings close\n            '
            embeddings = None
            for _ in range(2):
                embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'scoring': {'method': 'bm25', 'terms': True}, 'content': self.backend})
                embeddings.index([(0, 'Close test', None)])
                index = os.path.join(tempfile.gettempdir(), f'embeddings.{self.category()}.close')
                embeddings.save(index)
                embeddings.close()
            self.assertIsNone(embeddings.ann)
            self.assertIsNone(embeddings.database)

        def testData(self):
            if False:
                print('Hello World!')
            '\n            Test content storage and retrieval\n            '
            data = self.data + [{'date': '2021-01-01', 'text': 'Baby panda', 'flag': 1}]
            self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(data)])
            result = self.embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], data[-1]['text'])

        def testDelete(self):
            if False:
                i = 10
                return i + 15
            '\n            Test delete\n            '
            self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            self.embeddings.delete([4])
            result = self.embeddings.search('feel good story', 1)[0]
            self.assertEqual(self.embeddings.count(), 5)
            self.assertEqual(result['text'], self.data[5])

        def testEmpty(self):
            if False:
                while True:
                    i = 10
            '\n            Test empty index\n            '
            embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': self.backend})
            self.assertEqual(embeddings.search('test'), [])
            embeddings.index([])
            self.assertIsNone(embeddings.ann)
            embeddings.index([(0, 'this is a test', None)])
            embeddings.upsert([])
            self.assertIsNotNone(embeddings.ann)

        def testEmptyString(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test empty string indexing\n            '
            self.embeddings.index([(0, '', None)])
            self.assertTrue(self.embeddings.search('test'))
            self.embeddings.index([(0, {'text': ''}, None)])
            self.assertTrue(self.embeddings.search('test'))

        def testExplain(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test query explain\n            '
            result = self.embeddings.explain('feel good story', self.data)[0]
            self.assertEqual(result['text'], self.data[4])
            self.assertEqual(len(result.get('tokens')), 8)

        def testExplainBatch(self):
            if False:
                print('Hello World!')
            '\n            Test query explain batch\n            '
            self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            result = self.embeddings.batchexplain(['feel good story'], limit=1)[0][0]
            self.assertEqual(result['text'], self.data[4])
            self.assertEqual(len(result.get('tokens')), 8)

        def testExplainEmpty(self):
            if False:
                print('Hello World!')
            '\n            Test query explain with no filtering criteria\n            '
            self.assertEqual(self.embeddings.explain('select * from txtai limit 1')[0]['id'], '0')

        def testGenerator(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test index with a generator\n            '

            def documents():
                if False:
                    return 10
                for (uid, text) in enumerate(self.data):
                    yield (uid, text, None)
            self.embeddings.index(documents())
            result = self.embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])

        def testHybrid(self):
            if False:
                while True:
                    i = 10
            '\n            Test hybrid search\n            '
            data = [(uid, text, None) for (uid, text) in enumerate(self.data)]
            embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'hybrid': True, 'content': self.backend})
            embeddings.index(data)
            result = embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], data[4][1])
            index = os.path.join(tempfile.gettempdir(), f'embeddings.{self.category()}.hybrid')
            embeddings.save(index)
            embeddings.load(index)
            result = embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], data[4][1])
            embeddings.config['scoring']['normalize'] = False
            embeddings.index(data)
            result = embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], data[4][1])
            data[0] = (0, 'Feel good story: baby panda born', None)
            embeddings.upsert([data[0]])
            result = embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], data[0][1])

        def testIndex(self):
            if False:
                i = 10
                return i + 15
            '\n            Test index\n            '
            self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            result = self.embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])

        def testIndexTokens(self):
            if False:
                i = 10
                return i + 15
            '\n            Test index with tokens\n            '
            self.embeddings.index([(uid, text.split(), None) for (uid, text) in enumerate(self.data)])
            result = self.embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])

        def testInfo(self):
            if False:
                print('Hello World!')
            '\n            Test info\n            '
            self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.embeddings.info()
            self.assertIn('txtai', output.getvalue())

        def testInstructions(self):
            if False:
                print('Hello World!')
            '\n            Test indexing with instruction prefixes.\n            '
            embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': self.backend, 'instructions': {'query': 'query: ', 'data': 'passage: '}})
            embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            result = embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])

        def testInvalidData(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test invalid JSON data\n            '
            with self.assertRaises(ValueError):
                self.embeddings.index([(0, {'text': 'This is a test', 'flag': float('NaN')}, None)])

        def testJSON(self):
            if False:
                return 10
            '\n            Test JSON configuration\n            '
            embeddings = Embeddings({'format': 'json', 'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': self.backend})
            embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            index = os.path.join(tempfile.gettempdir(), f'embeddings.{self.category()}.json')
            embeddings.save(index)
            self.assertTrue(os.path.exists(os.path.join(index, 'config.json')))
            embeddings.load(index)
            self.assertEqual(embeddings.count(), 6)

        def testKeyword(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test keyword only (sparse) search\n            '
            data = [(uid, text, None) for (uid, text) in enumerate(self.data)]
            embeddings = Embeddings({'keyword': True, 'content': self.backend})
            embeddings.index(data)
            result = embeddings.search('lottery ticket', 1)[0]
            self.assertEqual(result['text'], data[4][1])
            self.assertEqual(embeddings.count(), len(data))
            index = os.path.join(tempfile.gettempdir(), f'embeddings.{self.category()}.keyword')
            embeddings.save(index)
            embeddings.load(index)
            result = embeddings.search('lottery ticket', 1)[0]
            self.assertEqual(result['text'], data[4][1])
            data[0] = (0, 'Feel good story: baby panda born', None)
            embeddings.upsert([data[0]])
            result = embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], data[0][1])

        def testMultiData(self):
            if False:
                while True:
                    i = 10
            '\n            Test indexing with multiple data types (text, documents)\n            '
            embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': self.backend, 'batch': len(self.data)})
            data = []
            for (uid, text) in enumerate(self.data):
                data.append((uid, text, None))
                data.append((uid, {'content': text}, None))
            embeddings.index(data)
            result = embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])

        def testMultiSave(self):
            if False:
                return 10
            '\n            Test multiple successive saves\n            '
            self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            index = os.path.join(tempfile.gettempdir(), f'embeddings.{self.category()}.insert')
            self.embeddings.save(index)
            self.embeddings.upsert([(0, 'Looking out into the dreadful abyss', None)])
            indexupdate = os.path.join(tempfile.gettempdir(), f'embeddings.{self.category()}.update')
            self.embeddings.save(indexupdate)
            self.embeddings.save(index)
            result = self.embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])
            self.embeddings.load(index)
            result = self.embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])
            self.embeddings.load(indexupdate)
            result = self.embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])

        def testNoIndex(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Tests an embeddings instance with no available indexes\n            '
            embeddings = Embeddings({'content': self.backend, 'defaults': False})
            embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            with self.assertRaises(IndexNotFoundError):
                embeddings.search("select id, text, score from txtai where similar('feel good story')")

        def testNotImplemented(self):
            if False:
                return 10
            '\n            Test exceptions for non-implemented methods\n            '
            db = RDBMS({})
            self.assertRaises(NotImplementedError, db.connect, None)
            self.assertRaises(NotImplementedError, db.getcursor)
            self.assertRaises(NotImplementedError, db.jsonprefix)
            self.assertRaises(NotImplementedError, db.jsoncolumn, None)
            self.assertRaises(NotImplementedError, db.rows)
            self.assertRaises(NotImplementedError, db.addfunctions)
            db = Embedded({})
            self.assertRaises(NotImplementedError, db.copy, None)

        def testObject(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test object field\n            '
            embeddings = Embeddings({'defaults': False, 'content': self.backend, 'objects': True})
            embeddings.index([{'object': 'binary data'.encode('utf-8')}])
            obj = embeddings.search('select object from txtai where id = 0')[0]['object']
            self.assertEqual(str(obj.getvalue(), 'utf-8'), 'binary data')

        def testQuantize(self):
            if False:
                i = 10
                return i + 15
            '\n            Test scalar quantization\n            '
            embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'quantize': 1, 'content': self.backend})
            embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            result = self.embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])

        def testQueryModel(self):
            if False:
                return 10
            '\n            Test index\n            '
            embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': self.backend, 'query': {'path': 'neuml/t5-small-txtsql'}})
            embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            result = embeddings.search('feel good story with win in text', 1)[0]
            self.assertEqual(result['text'], self.data[4])

        def testReindex(self):
            if False:
                while True:
                    i = 10
            '\n            Test reindex\n            '
            self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            self.embeddings.delete([0, 1])
            self.embeddings.reindex({'path': 'sentence-transformers/nli-mpnet-base-v2'})
            result = self.embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])

        def testSave(self):
            if False:
                return 10
            '\n            Test save\n            '
            self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            index = os.path.join(tempfile.gettempdir(), f'embeddings.{self.category()}')
            self.embeddings.save(index)
            self.embeddings.load(index)
            result = self.embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])
            self.embeddings.upsert([(0, 'Looking out into the dreadful abyss', None)])
            self.assertEqual(self.embeddings.count(), len(self.data))

        def testSettings(self):
            if False:
                while True:
                    i = 10
            '\n            Test custom SQLite settings\n            '
            embeddings = Embeddings({'path': 'sentence-transformers/nli-mpnet-base-v2', 'content': self.backend, 'sqlite': {'wal': True}})
            embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            result = embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], self.data[4])

        def testSQL(self):
            if False:
                i = 10
                return i + 15
            '\n            Test running a SQL query\n            '
            self.embeddings.index([(uid, {'text': text, 'length': len(text), 'attribute': f'ID{uid}'}, None) for (uid, text) in enumerate(self.data)])
            result = self.embeddings.search("select text, score from txtai where similar('feel good story') group by text, score having count(*) > 0 order by score desc", 1)[0]
            self.assertEqual(result['text'], self.data[4])
            result = self.embeddings.search("select * from txtai where similar('feel good story', 1) limit 1")[0]
            self.assertEqual(result['text'], self.data[4])
            result = self.embeddings.search("select * from txtai where similar('feel good story') offset 1")[0]
            self.assertEqual(result['text'], self.data[5])
            result = self.embeddings.search("select * from txtai where text like '%iceberg%'", 1)[0]
            self.assertEqual(result['text'], self.data[1])
            result = self.embeddings.search('select count(*) from txtai')[0]
            self.assertEqual(list(result.values())[0], len(self.data))
            result = self.embeddings.search('select id, text, length, data, entry from txtai')[0]
            self.assertEqual(sorted(result.keys()), ['data', 'entry', 'id', 'length', 'text'])
            result = self.embeddings.search("select text from txtai where attribute = 'ID4'", 1)[0]
            self.assertEqual(result['text'], self.data[4])
            with self.assertRaises(SQLError):
                self.embeddings.search('select * from txtai where bad,query')

        def testSQLBind(self):
            if False:
                return 10
            '\n            Test SQL statements with bind parameters\n            '
            self.embeddings.index([(uid, text, None) for (uid, text) in enumerate(self.data)])
            result = self.embeddings.search('select id, text, score from txtai where similar(:x)', parameters={'x': 'feel good story'})[0]
            self.assertEqual(result['text'], self.data[4])
            result = self.embeddings.search('select id, text, score from txtai where similar(:x, 0.5)', parameters={'x': 'feel good story'})[0]
            self.assertEqual(result['text'], self.data[4])
            result = self.embeddings.search('select * from txtai where text like :x', parameters={'x': '%iceberg%'})[0]
            self.assertEqual(result['text'], self.data[1])

        def testSubindex(self):
            if False:
                i = 10
                return i + 15
            '\n            Test subindex\n            '
            data = [(uid, text, None) for (uid, text) in enumerate(self.data)]
            embeddings = Embeddings({'content': self.backend, 'defaults': False, 'indexes': {'index1': {'path': 'sentence-transformers/nli-mpnet-base-v2'}}})
            embeddings.index(data)
            result = embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], data[4][1])
            result = embeddings.search("select id, text, score from txtai where similar('feel good story', 10, 0.5)")[0]
            self.assertEqual(result['text'], data[4][1])
            with self.assertRaises(IndexNotFoundError):
                embeddings.search("select id, text, score from txtai where similar('feel good story', 'notindex')")
            index = os.path.join(tempfile.gettempdir(), f'embeddings.{self.category()}.subindex')
            embeddings.save(index)
            embeddings.load(index)
            result = embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], data[4][1])
            data[0] = (0, 'Feel good story: baby panda born', None)
            embeddings.upsert([data[0]])
            result = embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], data[0][1])
            embeddings.upsert([(embeddings.count(), {'content': 'empty text'}, None)])
            result = embeddings.search(f'{embeddings.count() - 1}', 1)[0]
            self.assertEqual(result['text'], str(embeddings.count() - 1))
            embeddings.close()

        def testTerms(self):
            if False:
                return 10
            '\n            Test extracting keyword terms from query\n            '
            result = self.embeddings.terms("select * from txtai where similar('keyword terms')")
            self.assertEqual(result, 'keyword terms')

        def testUpsert(self):
            if False:
                i = 10
                return i + 15
            '\n            Test upsert\n            '
            data = [(uid, text, None) for (uid, text) in enumerate(self.data)]
            self.embeddings.ann = None
            self.embeddings.database = None
            self.embeddings.upsert(data)
            data[0] = (0, 'Feel good story: baby panda born', None)
            self.embeddings.upsert([data[0]])
            result = self.embeddings.search('feel good story', 1)[0]
            self.assertEqual(result['text'], data[0][1])

        def testUpsertBatch(self):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Test upsert batch\n            '
            try:
                data = [(uid, text, None) for (uid, text) in enumerate(self.data)]
                self.embeddings.ann = None
                self.embeddings.database = None
                self.embeddings.upsert(data)
                self.embeddings.config['batch'] = 1
                data[0] = (0, 'Feel good story: baby panda born', None)
                data[1] = (0, 'Not good news', None)
                self.embeddings.upsert([data[0], data[1]])
                result = self.embeddings.search('feel good story', 1)[0]
                self.assertEqual(result['text'], data[0][1])
            finally:
                del self.embeddings.config['batch']

        def category(self):
            if False:
                return 10
            '\n            Content backend category.\n\n            Returns:\n                category\n            '
            return self.__class__.__name__.lower().replace('test', '')