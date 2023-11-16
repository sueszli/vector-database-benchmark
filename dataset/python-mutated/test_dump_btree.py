"""Tests of the 'bzr dump-btree' command."""
from bzrlib import btree_index, tests
from bzrlib.tests import http_server

class TestDumpBtree(tests.TestCaseWithTransport):

    def create_sample_btree_index(self):
        if False:
            return 10
        builder = btree_index.BTreeBuilder(reference_lists=1, key_elements=2)
        builder.add_node(('test', 'key1'), 'value', ((('ref', 'entry'),),))
        builder.add_node(('test', 'key2'), 'value2', ((('ref', 'entry2'),),))
        builder.add_node(('test2', 'key3'), 'value3', ((('ref', 'entry3'),),))
        out_f = builder.finish()
        try:
            self.build_tree_contents([('test.btree', out_f.read())])
        finally:
            out_f.close()

    def test_dump_btree_smoke(self):
        if False:
            while True:
                i = 10
        self.create_sample_btree_index()
        (out, err) = self.run_bzr('dump-btree test.btree')
        self.assertEqualDiff("(('test', 'key1'), 'value', ((('ref', 'entry'),),))\n(('test', 'key2'), 'value2', ((('ref', 'entry2'),),))\n(('test2', 'key3'), 'value3', ((('ref', 'entry3'),),))\n", out)

    def test_dump_btree_http_smoke(self):
        if False:
            for i in range(10):
                print('nop')
        self.transport_readonly_server = http_server.HttpServer
        self.create_sample_btree_index()
        url = self.get_readonly_url('test.btree')
        (out, err) = self.run_bzr(['dump-btree', url])
        self.assertEqualDiff("(('test', 'key1'), 'value', ((('ref', 'entry'),),))\n(('test', 'key2'), 'value2', ((('ref', 'entry2'),),))\n(('test2', 'key3'), 'value3', ((('ref', 'entry3'),),))\n", out)

    def test_dump_btree_raw_smoke(self):
        if False:
            i = 10
            return i + 15
        self.create_sample_btree_index()
        (out, err) = self.run_bzr('dump-btree test.btree --raw')
        self.assertEqualDiff('Root node:\nB+Tree Graph Index 2\nnode_ref_lists=1\nkey_elements=2\nlen=3\nrow_lengths=1\n\nPage 0\ntype=leaf\ntest\x00key1\x00ref\x00entry\x00value\ntest\x00key2\x00ref\x00entry2\x00value2\ntest2\x00key3\x00ref\x00entry3\x00value3\n\n', out)

    def test_dump_btree_no_refs_smoke(self):
        if False:
            return 10
        builder = btree_index.BTreeBuilder(reference_lists=0, key_elements=2)
        builder.add_node(('test', 'key1'), 'value')
        out_f = builder.finish()
        try:
            self.build_tree_contents([('test.btree', out_f.read())])
        finally:
            out_f.close()
        (out, err) = self.run_bzr('dump-btree test.btree')

    def create_sample_empty_btree_index(self):
        if False:
            while True:
                i = 10
        builder = btree_index.BTreeBuilder(reference_lists=1, key_elements=2)
        out_f = builder.finish()
        try:
            self.build_tree_contents([('test.btree', out_f.read())])
        finally:
            out_f.close()

    def test_dump_empty_btree_smoke(self):
        if False:
            for i in range(10):
                print('nop')
        self.create_sample_empty_btree_index()
        (out, err) = self.run_bzr('dump-btree test.btree')
        self.assertEqualDiff('', out)

    def test_dump_empty_btree_http_smoke(self):
        if False:
            i = 10
            return i + 15
        self.transport_readonly_server = http_server.HttpServer
        self.create_sample_empty_btree_index()
        url = self.get_readonly_url('test.btree')
        (out, err) = self.run_bzr(['dump-btree', url])
        self.assertEqualDiff('', out)

    def test_dump_empty_btree_raw_smoke(self):
        if False:
            print('Hello World!')
        self.create_sample_empty_btree_index()
        (out, err) = self.run_bzr('dump-btree test.btree --raw')
        self.assertEqualDiff('Root node:\nB+Tree Graph Index 2\nnode_ref_lists=1\nkey_elements=2\nlen=0\nrow_lengths=\n\nPage 0\n(empty)\n', out)