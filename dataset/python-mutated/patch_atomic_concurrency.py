import sys
import simplejson as json
import eve.methods.common
from eve.utils import config
from tests import TestBase
'\nAtomic Concurrency Checks\n\nPrior to commit 54fd697 from 2016-November, ETags would be verified\ntwice during a patch. One ETag check would be non-atomic by Eve,\nthen again atomically by MongoDB during app.data.update(filter).\nThe atomic ETag check was removed during issue #920 in 54fd697\n\nWhen running Eve in a scale-out environment (multiple processes),\nconcurrent simultaneous updates are sometimes allowed, because\nthe Python-only ETag check is not atomic.\n\nThere is a critical section in patch_internal() between get_document()\nand app.data.update() where a competing Eve process can change the\ndocument and ETag.\n\nThis test simulates another process changing data & ETag during\nthe critical section. The test patches get_document() to return an\nintentionally wrong ETag.\n'

def get_document_simulate_concurrent_update(*args, **kwargs):
    if False:
        print('Hello World!')
    '\n    Hostile version of get_document\n\n    This simluates another process updating MongoDB (and ETag) in\n    eve.methods.patch.patch_internal() during the critical area\n    between get_document() and app.data.update()\n    '
    document = eve.methods.common.get_document(*args, **kwargs)
    document[config.ETAG] = 'unexpected change!'
    return document

class TestPatchAtomicConcurrent(TestBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Patch eve.methods.patch.get_document with a hostile version\n        that simulates simultaneous updates\n        '
        self.original_get_document = sys.modules['eve.methods.patch'].get_document
        sys.modules['eve.methods.patch'].get_document = get_document_simulate_concurrent_update
        return super().setUp()

    def test_etag_changed_after_get_document(self):
        if False:
            while True:
                i = 10
        '\n        Try to update a document after the ETag was adjusted\n        outside this process\n        '
        changes = {'ref': '1234567890123456789054321'}
        (_r, status) = self.patch(self.item_id_url, data=changes, headers=[('If-Match', self.item_etag)])
        self.assertEqual(status, 412)

    def tearDown(self):
        if False:
            return 10
        'Remove patch of eve.methods.patch.get_document'
        sys.modules['eve.methods.patch'].get_document = self.original_get_document
        return super().tearDown()

    def patch(self, url, data, headers=[]):
        if False:
            print('Hello World!')
        headers.append(('Content-Type', 'application/json'))
        r = self.test_client.patch(url, data=json.dumps(data), headers=headers)
        return self.parse_response(r)