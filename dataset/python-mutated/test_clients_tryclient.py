import base64
import json
import sys
from twisted.trial import unittest
from buildbot.clients import tryclient
from buildbot.util import bytes2unicode

class createJobfile(unittest.TestCase):

    def makeNetstring(self, *strings):
        if False:
            i = 10
            return i + 15
        return ''.join([f'{len(s)}:{s},' for s in strings])

    def test_createJobfile_v5(self):
        if False:
            print('Hello World!')
        jobid = '123-456'
        branch = 'branch'
        baserev = 'baserev'
        patch_level = 0
        patch_body = b'diff...'
        repository = 'repo'
        project = 'proj'
        who = 'someuser'
        comment = 'insightful comment'
        builderNames = ['runtests']
        properties = {'foo': 'bar'}
        job = tryclient.createJobfile(jobid, branch, baserev, patch_level, patch_body, repository, project, who, comment, builderNames, properties)
        jobstr = self.makeNetstring('5', json.dumps({'jobid': jobid, 'branch': branch, 'baserev': baserev, 'patch_level': patch_level, 'repository': repository, 'project': project, 'who': who, 'comment': comment, 'builderNames': builderNames, 'properties': properties, 'patch_body': bytes2unicode(patch_body)}))
        self.assertEqual(job, jobstr)

    def test_createJobfile_v6(self):
        if False:
            i = 10
            return i + 15
        jobid = '123-456'
        branch = 'branch'
        baserev = 'baserev'
        patch_level = 0
        patch_body = b'diff...\xff'
        repository = 'repo'
        project = 'proj'
        who = 'someuser'
        comment = 'insightful comment'
        builderNames = ['runtests']
        properties = {'foo': 'bar'}
        job = tryclient.createJobfile(jobid, branch, baserev, patch_level, patch_body, repository, project, who, comment, builderNames, properties)
        jobstr = self.makeNetstring('6', json.dumps({'jobid': jobid, 'branch': branch, 'baserev': baserev, 'patch_level': patch_level, 'repository': repository, 'project': project, 'who': who, 'comment': comment, 'builderNames': builderNames, 'properties': properties, 'patch_body_base64': bytes2unicode(base64.b64encode(patch_body))}))
        self.assertEqual(job, jobstr)

    def test_SourceStampExtractor_readPatch(self):
        if False:
            for i in range(10):
                print('nop')
        sse = tryclient.GitExtractor(None, None, None)
        for (patchlevel, diff) in enumerate((None, '', b'')):
            sse.readPatch(diff, patchlevel)
            self.assertEqual(sse.patch, (patchlevel, None))
        sse.readPatch(b'diff schmiff blah blah blah', 23)
        self.assertEqual(sse.patch, (23, b'diff schmiff blah blah blah'))

    def test_GitExtractor_fixBranch(self):
        if False:
            return 10
        sse = tryclient.GitExtractor(None, 'origin/master', None)
        self.assertEqual(sse.branch, 'origin/master')
        sse.fixBranch(b'origi\n')
        self.assertEqual(sse.branch, 'origin/master')
        sse.fixBranch(b'origin\n')
        self.assertEqual(sse.branch, 'master')

    def test_GitExtractor_override_baserev(self):
        if False:
            while True:
                i = 10
        sse = tryclient.GitExtractor(None, None, None)
        sse.override_baserev(b'23ae367063327b79234e081f396ecbc\n')
        self.assertEqual(sse.baserev, '23ae367063327b79234e081f396ecbc')

    class RemoteTryPP_TestStream:

        def __init__(self):
            if False:
                print('Hello World!')
            self.writes = []
            self.is_open = True

        def write(self, data):
            if False:
                print('Hello World!')
            assert self.is_open
            self.writes.append(data)

        def closeStdin(self):
            if False:
                print('Hello World!')
            assert self.is_open
            self.is_open = False

    def test_RemoteTryPP_encoding(self):
        if False:
            print('Hello World!')
        rmt = tryclient.RemoteTryPP('job')
        self.assertTrue(isinstance(rmt.job, str))
        rmt.transport = self.RemoteTryPP_TestStream()
        rmt.connectionMade()
        self.assertFalse(rmt.transport.is_open)
        self.assertEqual(len(rmt.transport.writes), 1)
        self.assertFalse(isinstance(rmt.transport.writes[0], str))
        for streamname in ('out', 'err'):
            sys_streamattr = 'std' + streamname
            rmt_methodattr = streamname + 'Received'
            teststream = self.RemoteTryPP_TestStream()
            saved_stream = getattr(sys, sys_streamattr)
            try:
                setattr(sys, sys_streamattr, teststream)
                getattr(rmt, rmt_methodattr)(b'data')
            finally:
                setattr(sys, sys_streamattr, saved_stream)
            self.assertEqual(len(teststream.writes), 1)
            self.assertTrue(isinstance(teststream.writes[0], str))