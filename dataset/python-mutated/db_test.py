from caffe2.python import workspace
import os
import tempfile
import unittest

class TestDB(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        (handle, self.file_name) = tempfile.mkstemp()
        os.close(handle)
        self.data = [('key{}'.format(i).encode('ascii'), 'value{}'.format(i).encode('ascii')) for i in range(1, 10)]

    def testSimple(self):
        if False:
            print('Hello World!')
        db = workspace.C.create_db('minidb', self.file_name, workspace.C.Mode.write)
        for (key, value) in self.data:
            transaction = db.new_transaction()
            transaction.put(key, value)
            del transaction
        del db
        db = workspace.C.create_db('minidb', self.file_name, workspace.C.Mode.read)
        cursor = db.new_cursor()
        data = []
        while cursor.valid():
            data.append((cursor.key(), cursor.value()))
            cursor.next()
        del cursor
        db.close()
        self.assertEqual(data, self.data)