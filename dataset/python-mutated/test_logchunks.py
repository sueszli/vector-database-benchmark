import textwrap
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.data import logchunks
from buildbot.data import resultspec
from buildbot.test import fakedb
from buildbot.test.util import endpoint

class LogChunkEndpointBase(endpoint.EndpointMixin, unittest.TestCase):
    endpointClass = logchunks.LogChunkEndpoint
    resourceTypeClass = logchunks.LogChunk
    endpointname = 'contents'
    log60Lines = ['line zero', 'line 1', 'line TWO', 'line 3', 'line 2**2', 'another line', 'yet another line']
    log61Lines = [f'{i:08d}' for i in range(100)]

    def setUp(self):
        if False:
            return 10
        self.setUpEndpoint()
        self.db.insert_test_data([fakedb.Builder(id=77), fakedb.Worker(id=13, name='wrk'), fakedb.Master(id=88), fakedb.Buildset(id=8822), fakedb.BuildRequest(id=82, buildsetid=8822), fakedb.Build(id=13, builderid=77, masterid=88, workerid=13, buildrequestid=82, number=3), fakedb.Step(id=50, buildid=13, number=9, name='make'), fakedb.Log(id=60, stepid=50, name='stdio', slug='stdio', type='s', num_lines=7), fakedb.LogChunk(logid=60, first_line=0, last_line=1, compressed=0, content=textwrap.dedent('                        line zero\n                        line 1')), fakedb.LogChunk(logid=60, first_line=2, last_line=4, compressed=0, content=textwrap.dedent('                        line TWO\n                        line 3\n                        line 2**2')), fakedb.LogChunk(logid=60, first_line=5, last_line=5, compressed=0, content='another line'), fakedb.LogChunk(logid=60, first_line=6, last_line=6, compressed=0, content='yet another line'), fakedb.Log(id=61, stepid=50, name='errors', slug='errors', type='t', num_lines=100)] + [fakedb.LogChunk(logid=61, first_line=i, last_line=i, compressed=0, content=f'{i:08d}') for i in range(100)] + [fakedb.Log(id=62, stepid=50, name='notes', slug='notes', type='t', num_lines=0)])

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tearDownEndpoint()

    @defer.inlineCallbacks
    def do_test_chunks(self, path, logid, expLines):
        if False:
            return 10
        logchunk = (yield self.callGet(path))
        self.validateData(logchunk)
        expContent = '\n'.join(expLines) + '\n'
        self.assertEqual(logchunk, {'logid': logid, 'firstline': 0, 'content': expContent})
        for (i, expLine) in enumerate(expLines):
            logchunk = (yield self.callGet(path, resultSpec=resultspec.ResultSpec(offset=i, limit=1)))
            self.validateData(logchunk)
            self.assertEqual(logchunk, {'logid': logid, 'firstline': i, 'content': expLine + '\n'})
        mid = int(len(expLines) / 2)
        for (f, length) in ((0, mid), (mid, len(expLines) - 1)):
            result_spec = resultspec.ResultSpec(offset=f, limit=length - f + 1)
            logchunk = (yield self.callGet(path, resultSpec=result_spec))
            self.validateData(logchunk)
            expContent = '\n'.join(expLines[f:length + 1]) + '\n'
            self.assertEqual(logchunk, {'logid': logid, 'firstline': f, 'content': expContent})
        (f, length) = (len(expLines) - 2, len(expLines) + 10)
        result_spec = resultspec.ResultSpec(offset=f, limit=length - f + 1)
        logchunk = (yield self.callGet(path, resultSpec=result_spec))
        self.validateData(logchunk)
        expContent = '\n'.join(expLines[-2:]) + '\n'
        self.assertEqual(logchunk, {'logid': logid, 'firstline': f, 'content': expContent})
        self.assertEqual((yield self.callGet(path, resultSpec=resultspec.ResultSpec(offset=-1))), None)
        self.assertEqual((yield self.callGet(path, resultSpec=resultspec.ResultSpec(offset=10, limit=-1))), None)

    def test_get_logid_60(self):
        if False:
            i = 10
            return i + 15
        return self.do_test_chunks(('logs', 60, self.endpointname), 60, self.log60Lines)

    def test_get_logid_61(self):
        if False:
            for i in range(10):
                print('nop')
        return self.do_test_chunks(('logs', 61, self.endpointname), 61, self.log61Lines)

class LogChunkEndpoint(LogChunkEndpointBase):

    @defer.inlineCallbacks
    def test_get_missing(self):
        if False:
            for i in range(10):
                print('nop')
        logchunk = (yield self.callGet(('logs', 99, self.endpointname)))
        self.assertEqual(logchunk, None)

    @defer.inlineCallbacks
    def test_get_empty(self):
        if False:
            return 10
        logchunk = (yield self.callGet(('logs', 62, self.endpointname)))
        self.validateData(logchunk)
        self.assertEqual(logchunk['content'], '')

    @defer.inlineCallbacks
    def test_get_by_stepid(self):
        if False:
            for i in range(10):
                print('nop')
        logchunk = (yield self.callGet(('steps', 50, 'logs', 'errors', self.endpointname)))
        self.validateData(logchunk)
        self.assertEqual(logchunk['logid'], 61)

    @defer.inlineCallbacks
    def test_get_by_buildid(self):
        if False:
            print('Hello World!')
        logchunk = (yield self.callGet(('builds', 13, 'steps', 9, 'logs', 'stdio', self.endpointname)))
        self.validateData(logchunk)
        self.assertEqual(logchunk['logid'], 60)

    @defer.inlineCallbacks
    def test_get_by_builder(self):
        if False:
            i = 10
            return i + 15
        logchunk = (yield self.callGet(('builders', 77, 'builds', 3, 'steps', 9, 'logs', 'errors', self.endpointname)))
        self.validateData(logchunk)
        self.assertEqual(logchunk['logid'], 61)

    @defer.inlineCallbacks
    def test_get_by_builder_step_name(self):
        if False:
            for i in range(10):
                print('nop')
        logchunk = (yield self.callGet(('builders', 77, 'builds', 3, 'steps', 'make', 'logs', 'errors', self.endpointname)))
        self.validateData(logchunk)
        self.assertEqual(logchunk['logid'], 61)

class RawLogChunkEndpoint(LogChunkEndpointBase):
    endpointClass = logchunks.RawLogChunkEndpoint
    endpointname = 'raw'

    def validateData(self, data):
        if False:
            i = 10
            return i + 15
        self.assertIsInstance(data['raw'], str)
        self.assertIsInstance(data['mime-type'], str)
        self.assertIsInstance(data['filename'], str)

    @defer.inlineCallbacks
    def do_test_chunks(self, path, logid, expLines):
        if False:
            print('Hello World!')
        logchunk = (yield self.callGet(path))
        self.validateData(logchunk)
        if logid == 60:
            expContent = '\n'.join([line[1:] for line in expLines])
            expFilename = 'stdio'
        else:
            expContent = '\n'.join(expLines) + '\n'
            expFilename = 'errors'
        self.assertEqual(logchunk, {'filename': expFilename, 'mime-type': 'text/plain', 'raw': expContent})