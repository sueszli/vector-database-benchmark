import base64
import json
import requests_mock
from mitmproxy.test import taddons
from mitmproxy.test import tflow
from mitmproxy.test import tutils
example_dir = tutils.test_data.push('../examples')

class TestJSONDump:

    def echo_response(self, request, context):
        if False:
            while True:
                i = 10
        self.request = {'json': request.json(), 'headers': request.headers}
        return ''

    def flow(self, resp_content=b'message'):
        if False:
            print('Hello World!')
        times = dict(timestamp_start=746203272, timestamp_end=746203272)
        return tflow.tflow(req=tutils.treq(method=b'GET', **times), resp=tutils.tresp(content=resp_content, **times))

    def test_simple(self, tmpdir):
        if False:
            for i in range(10):
                print('nop')
        with taddons.context() as tctx:
            a = tctx.script(example_dir.path('complex/jsondump.py'))
            path = str(tmpdir.join('jsondump.out'))
            tctx.configure(a, dump_destination=path)
            tctx.invoke(a, 'response', self.flow())
            tctx.invoke(a, 'done')
            with open(path) as inp:
                entry = json.loads(inp.readline())
            assert entry['response']['content'] == 'message'

    def test_contentencode(self, tmpdir):
        if False:
            while True:
                i = 10
        with taddons.context() as tctx:
            a = tctx.script(example_dir.path('complex/jsondump.py'))
            path = str(tmpdir.join('jsondump.out'))
            content = b'foo' + b'\xff' * 10
            tctx.configure(a, dump_destination=path, dump_encodecontent=True)
            tctx.invoke(a, 'response', self.flow(resp_content=content))
            tctx.invoke(a, 'done')
            with open(path) as inp:
                entry = json.loads(inp.readline())
            assert entry['response']['content'] == base64.b64encode(content).decode('utf-8')

    def test_http(self, tmpdir):
        if False:
            for i in range(10):
                print('nop')
        with requests_mock.Mocker() as mock:
            mock.post('http://my-server', text=self.echo_response)
            with taddons.context() as tctx:
                a = tctx.script(example_dir.path('complex/jsondump.py'))
                tctx.configure(a, dump_destination='http://my-server', dump_username='user', dump_password='pass')
                tctx.invoke(a, 'response', self.flow())
                tctx.invoke(a, 'done')
                assert self.request['json']['response']['content'] == 'message'
                assert self.request['headers']['Authorization'] == 'Basic dXNlcjpwYXNz'