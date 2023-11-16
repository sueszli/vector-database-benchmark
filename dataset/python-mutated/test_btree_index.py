"""Tests for btree indices."""
import pprint
import zlib
from bzrlib import btree_index, errors, fifo_cache, lru_cache, osutils, tests, transport
from bzrlib.tests import TestCaseWithTransport, scenarios
from bzrlib.tests import features
load_tests = scenarios.load_tests_apply_scenarios

def btreeparser_scenarios():
    if False:
        while True:
            i = 10
    import bzrlib._btree_serializer_py as py_module
    scenarios = [('python', {'parse_btree': py_module})]
    if compiled_btreeparser_feature.available():
        scenarios.append(('C', {'parse_btree': compiled_btreeparser_feature.module}))
    return scenarios
compiled_btreeparser_feature = features.ModuleAvailableFeature('bzrlib._btree_serializer_pyx')

class BTreeTestCase(TestCaseWithTransport):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(BTreeTestCase, self).setUp()
        self.overrideAttr(btree_index, '_RESERVED_HEADER_BYTES', 100)

    def make_nodes(self, count, key_elements, reference_lists):
        if False:
            for i in range(10):
                print('nop')
        'Generate count*key_elements sample nodes.'
        keys = []
        for prefix_pos in range(key_elements):
            if key_elements - 1:
                prefix = (str(prefix_pos) * 40,)
            else:
                prefix = ()
            for pos in xrange(count):
                key = prefix + (str(pos) * 40,)
                value = 'value:%s' % pos
                if reference_lists:
                    refs = []
                    for list_pos in range(reference_lists):
                        refs.append([])
                        for ref_pos in range(list_pos + pos % 2):
                            if pos % 2:
                                refs[-1].append(prefix + ('ref' + str(pos - 1) * 40,))
                            else:
                                refs[-1].append(prefix + ('ref' + str(ref_pos) * 40,))
                        refs[-1] = tuple(refs[-1])
                    refs = tuple(refs)
                else:
                    refs = ()
                keys.append((key, value, refs))
        return keys

    def shrink_page_size(self):
        if False:
            i = 10
            return i + 15
        'Shrink the default page size so that less fits in a page.'
        self.overrideAttr(btree_index, '_PAGE_SIZE')
        btree_index._PAGE_SIZE = 2048

    def assertEqualApproxCompressed(self, expected, actual, slop=6):
        if False:
            for i in range(10):
                print('nop')
        'Check a count of compressed bytes is approximately as expected\n\n        Relying on compressed length being stable even with fixed inputs is\n        slightly bogus, but zlib is stable enough that this mostly works.\n        '
        if not expected - slop < actual < expected + slop:
            self.fail('Expected around %d bytes compressed but got %d' % (expected, actual))

class TestBTreeBuilder(BTreeTestCase):

    def test_clear_cache(self):
        if False:
            i = 10
            return i + 15
        builder = btree_index.BTreeBuilder(reference_lists=0, key_elements=1)
        builder.clear_cache()

    def test_empty_1_0(self):
        if False:
            while True:
                i = 10
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqual('B+Tree Graph Index 2\nnode_ref_lists=0\nkey_elements=1\nlen=0\nrow_lengths=\n', content)

    def test_empty_2_1(self):
        if False:
            for i in range(10):
                print('nop')
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=1)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqual('B+Tree Graph Index 2\nnode_ref_lists=1\nkey_elements=2\nlen=0\nrow_lengths=\n', content)

    def test_root_leaf_1_0(self):
        if False:
            print('Hello World!')
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        nodes = self.make_nodes(5, 1, 0)
        for node in nodes:
            builder.add_node(*node)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqual(131, len(content))
        self.assertEqual('B+Tree Graph Index 2\nnode_ref_lists=0\nkey_elements=1\nlen=5\nrow_lengths=1\n', content[:73])
        node_content = content[73:]
        node_bytes = zlib.decompress(node_content)
        expected_node = 'type=leaf\n0000000000000000000000000000000000000000\x00\x00value:0\n1111111111111111111111111111111111111111\x00\x00value:1\n2222222222222222222222222222222222222222\x00\x00value:2\n3333333333333333333333333333333333333333\x00\x00value:3\n4444444444444444444444444444444444444444\x00\x00value:4\n'
        self.assertEqual(expected_node, node_bytes)

    def test_root_leaf_2_2(self):
        if False:
            for i in range(10):
                print('nop')
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(5, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqual(238, len(content))
        self.assertEqual('B+Tree Graph Index 2\nnode_ref_lists=2\nkey_elements=2\nlen=10\nrow_lengths=1\n', content[:74])
        node_content = content[74:]
        node_bytes = zlib.decompress(node_content)
        expected_node = 'type=leaf\n0000000000000000000000000000000000000000\x000000000000000000000000000000000000000000\x00\t0000000000000000000000000000000000000000\x00ref0000000000000000000000000000000000000000\x00value:0\n0000000000000000000000000000000000000000\x001111111111111111111111111111111111111111\x000000000000000000000000000000000000000000\x00ref0000000000000000000000000000000000000000\t0000000000000000000000000000000000000000\x00ref0000000000000000000000000000000000000000\r0000000000000000000000000000000000000000\x00ref0000000000000000000000000000000000000000\x00value:1\n0000000000000000000000000000000000000000\x002222222222222222222222222222222222222222\x00\t0000000000000000000000000000000000000000\x00ref0000000000000000000000000000000000000000\x00value:2\n0000000000000000000000000000000000000000\x003333333333333333333333333333333333333333\x000000000000000000000000000000000000000000\x00ref2222222222222222222222222222222222222222\t0000000000000000000000000000000000000000\x00ref2222222222222222222222222222222222222222\r0000000000000000000000000000000000000000\x00ref2222222222222222222222222222222222222222\x00value:3\n0000000000000000000000000000000000000000\x004444444444444444444444444444444444444444\x00\t0000000000000000000000000000000000000000\x00ref0000000000000000000000000000000000000000\x00value:4\n1111111111111111111111111111111111111111\x000000000000000000000000000000000000000000\x00\t1111111111111111111111111111111111111111\x00ref0000000000000000000000000000000000000000\x00value:0\n1111111111111111111111111111111111111111\x001111111111111111111111111111111111111111\x001111111111111111111111111111111111111111\x00ref0000000000000000000000000000000000000000\t1111111111111111111111111111111111111111\x00ref0000000000000000000000000000000000000000\r1111111111111111111111111111111111111111\x00ref0000000000000000000000000000000000000000\x00value:1\n1111111111111111111111111111111111111111\x002222222222222222222222222222222222222222\x00\t1111111111111111111111111111111111111111\x00ref0000000000000000000000000000000000000000\x00value:2\n1111111111111111111111111111111111111111\x003333333333333333333333333333333333333333\x001111111111111111111111111111111111111111\x00ref2222222222222222222222222222222222222222\t1111111111111111111111111111111111111111\x00ref2222222222222222222222222222222222222222\r1111111111111111111111111111111111111111\x00ref2222222222222222222222222222222222222222\x00value:3\n1111111111111111111111111111111111111111\x004444444444444444444444444444444444444444\x00\t1111111111111111111111111111111111111111\x00ref0000000000000000000000000000000000000000\x00value:4\n'
        self.assertEqual(expected_node, node_bytes)

    def test_2_leaves_1_0(self):
        if False:
            return 10
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        nodes = self.make_nodes(400, 1, 0)
        for node in nodes:
            builder.add_node(*node)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqualApproxCompressed(9283, len(content))
        self.assertEqual('B+Tree Graph Index 2\nnode_ref_lists=0\nkey_elements=1\nlen=400\nrow_lengths=1,2\n', content[:77])
        root = content[77:4096]
        leaf1 = content[4096:8192]
        leaf2 = content[8192:]
        root_bytes = zlib.decompress(root)
        expected_root = 'type=internal\noffset=0\n' + '307' * 40 + '\n'
        self.assertEqual(expected_root, root_bytes)
        leaf1_bytes = zlib.decompress(leaf1)
        sorted_node_keys = sorted((node[0] for node in nodes))
        node = btree_index._LeafNode(leaf1_bytes, 1, 0)
        self.assertEqual(231, len(node))
        self.assertEqual(sorted_node_keys[:231], node.all_keys())
        leaf2_bytes = zlib.decompress(leaf2)
        node = btree_index._LeafNode(leaf2_bytes, 1, 0)
        self.assertEqual(400 - 231, len(node))
        self.assertEqual(sorted_node_keys[231:], node.all_keys())

    def test_last_page_rounded_1_layer(self):
        if False:
            print('Hello World!')
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        nodes = self.make_nodes(10, 1, 0)
        for node in nodes:
            builder.add_node(*node)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqualApproxCompressed(155, len(content))
        self.assertEqual('B+Tree Graph Index 2\nnode_ref_lists=0\nkey_elements=1\nlen=10\nrow_lengths=1\n', content[:74])
        leaf2 = content[74:]
        leaf2_bytes = zlib.decompress(leaf2)
        node = btree_index._LeafNode(leaf2_bytes, 1, 0)
        self.assertEqual(10, len(node))
        sorted_node_keys = sorted((node[0] for node in nodes))
        self.assertEqual(sorted_node_keys, node.all_keys())

    def test_last_page_not_rounded_2_layer(self):
        if False:
            while True:
                i = 10
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        nodes = self.make_nodes(400, 1, 0)
        for node in nodes:
            builder.add_node(*node)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqualApproxCompressed(9283, len(content))
        self.assertEqual('B+Tree Graph Index 2\nnode_ref_lists=0\nkey_elements=1\nlen=400\nrow_lengths=1,2\n', content[:77])
        leaf2 = content[8192:]
        leaf2_bytes = zlib.decompress(leaf2)
        node = btree_index._LeafNode(leaf2_bytes, 1, 0)
        self.assertEqual(400 - 231, len(node))
        sorted_node_keys = sorted((node[0] for node in nodes))
        self.assertEqual(sorted_node_keys[231:], node.all_keys())

    def test_three_level_tree_details(self):
        if False:
            return 10
        self.shrink_page_size()
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(20000, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', self.time(builder.finish))
        del builder
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        index.key_count()
        self.assertEqual(3, len(index._row_lengths), 'Not enough rows: %r' % index._row_lengths)
        self.assertEqual(4, len(index._row_offsets))
        self.assertEqual(sum(index._row_lengths), index._row_offsets[-1])
        internal_nodes = index._get_internal_nodes([0, 1, 2])
        root_node = internal_nodes[0]
        internal_node1 = internal_nodes[1]
        internal_node2 = internal_nodes[2]
        self.assertEqual(internal_node2.offset, 1 + len(internal_node1.keys))
        pos = index._row_offsets[2] + internal_node2.offset + 1
        leaf = index._get_leaf_nodes([pos])[pos]
        self.assertTrue(internal_node2.keys[0] in leaf)

    def test_2_leaves_2_2(self):
        if False:
            for i in range(10):
                print('nop')
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(100, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqualApproxCompressed(12643, len(content))
        self.assertEqual('B+Tree Graph Index 2\nnode_ref_lists=2\nkey_elements=2\nlen=200\nrow_lengths=1,3\n', content[:77])
        root = content[77:4096]
        leaf1 = content[4096:8192]
        leaf2 = content[8192:12288]
        leaf3 = content[12288:]
        root_bytes = zlib.decompress(root)
        expected_root = 'type=internal\noffset=0\n' + '0' * 40 + '\x00' + '91' * 40 + '\n' + '1' * 40 + '\x00' + '81' * 40 + '\n'
        self.assertEqual(expected_root, root_bytes)

    def test_spill_index_stress_1_1(self):
        if False:
            return 10
        builder = btree_index.BTreeBuilder(key_elements=1, spill_at=2)
        nodes = [node[0:2] for node in self.make_nodes(16, 1, 0)]
        builder.add_node(*nodes[0])
        self.assertEqual(1, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        builder.add_node(*nodes[1])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(1, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        builder.add_node(*nodes[2])
        self.assertEqual(1, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        builder.add_node(*nodes[3])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(2, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(4, builder._backing_indices[1].key_count())
        builder.add_node(*nodes[4])
        builder.add_node(*nodes[5])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(2, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        self.assertEqual(4, builder._backing_indices[1].key_count())
        builder.add_node(*nodes[6])
        builder.add_node(*nodes[7])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(None, builder._backing_indices[1])
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[8])
        builder.add_node(*nodes[9])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        self.assertEqual(None, builder._backing_indices[1])
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[10])
        builder.add_node(*nodes[11])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(4, builder._backing_indices[1].key_count())
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[12])
        self.assertEqual([(builder,) + node for node in sorted(nodes[:13])], list(builder.iter_all_entries()))
        self.assertEqual(set([(builder,) + node for node in nodes[11:13]]), set(builder.iter_entries([nodes[12][0], nodes[11][0]])))
        self.assertEqual(13, builder.key_count())
        self.assertEqual(set([(builder,) + node for node in nodes[11:13]]), set(builder.iter_entries_prefix([nodes[12][0], nodes[11][0]])))
        builder.add_node(*nodes[13])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        self.assertEqual(4, builder._backing_indices[1].key_count())
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[14])
        builder.add_node(*nodes[15])
        self.assertEqual(4, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(None, builder._backing_indices[1])
        self.assertEqual(None, builder._backing_indices[2])
        self.assertEqual(16, builder._backing_indices[3].key_count())
        t = self.get_transport('')
        size = t.put_file('index', builder.finish())
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        nodes = list(index.iter_all_entries())
        self.assertEqual(sorted(nodes), nodes)
        self.assertEqual(16, len(nodes))

    def test_spill_index_stress_1_1_no_combine(self):
        if False:
            print('Hello World!')
        builder = btree_index.BTreeBuilder(key_elements=1, spill_at=2)
        builder.set_optimize(for_size=False, combine_backing_indices=False)
        nodes = [node[0:2] for node in self.make_nodes(16, 1, 0)]
        builder.add_node(*nodes[0])
        self.assertEqual(1, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        builder.add_node(*nodes[1])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(1, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        builder.add_node(*nodes[2])
        self.assertEqual(1, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        builder.add_node(*nodes[3])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(2, len(builder._backing_indices))
        for backing_index in builder._backing_indices:
            self.assertEqual(2, backing_index.key_count())
        builder.add_node(*nodes[4])
        builder.add_node(*nodes[5])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(3, len(builder._backing_indices))
        for backing_index in builder._backing_indices:
            self.assertEqual(2, backing_index.key_count())
        builder.add_node(*nodes[6])
        builder.add_node(*nodes[7])
        builder.add_node(*nodes[8])
        builder.add_node(*nodes[9])
        builder.add_node(*nodes[10])
        builder.add_node(*nodes[11])
        builder.add_node(*nodes[12])
        self.assertEqual(6, len(builder._backing_indices))
        for backing_index in builder._backing_indices:
            self.assertEqual(2, backing_index.key_count())
        self.assertEqual([(builder,) + node for node in sorted(nodes[:13])], list(builder.iter_all_entries()))
        self.assertEqual(set([(builder,) + node for node in nodes[11:13]]), set(builder.iter_entries([nodes[12][0], nodes[11][0]])))
        self.assertEqual(13, builder.key_count())
        self.assertEqual(set([(builder,) + node for node in nodes[11:13]]), set(builder.iter_entries_prefix([nodes[12][0], nodes[11][0]])))
        builder.add_node(*nodes[13])
        builder.add_node(*nodes[14])
        builder.add_node(*nodes[15])
        self.assertEqual(8, len(builder._backing_indices))
        for backing_index in builder._backing_indices:
            self.assertEqual(2, backing_index.key_count())
        transport = self.get_transport('')
        size = transport.put_file('index', builder.finish())
        index = btree_index.BTreeGraphIndex(transport, 'index', size)
        nodes = list(index.iter_all_entries())
        self.assertEqual(sorted(nodes), nodes)
        self.assertEqual(16, len(nodes))

    def test_set_optimize(self):
        if False:
            return 10
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        builder.set_optimize(for_size=True)
        self.assertTrue(builder._optimize_for_size)
        builder.set_optimize(for_size=False)
        self.assertFalse(builder._optimize_for_size)
        obj = object()
        builder._optimize_for_size = obj
        builder.set_optimize(combine_backing_indices=False)
        self.assertFalse(builder._combine_backing_indices)
        self.assertIs(obj, builder._optimize_for_size)
        builder.set_optimize(combine_backing_indices=True)
        self.assertTrue(builder._combine_backing_indices)
        self.assertIs(obj, builder._optimize_for_size)

    def test_spill_index_stress_2_2(self):
        if False:
            for i in range(10):
                print('nop')
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2, spill_at=2)
        nodes = self.make_nodes(16, 2, 2)
        builder.add_node(*nodes[0])
        self.assertEqual(1, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        builder.add_node(*nodes[1])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(1, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        old = dict(builder._get_nodes_by_key())
        builder.add_node(*nodes[2])
        self.assertEqual(1, len(builder._nodes))
        self.assertIsNot(None, builder._nodes_by_key)
        self.assertNotEqual({}, builder._nodes_by_key)
        self.assertNotEqual(old, builder._nodes_by_key)
        builder.add_node(*nodes[3])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(2, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(4, builder._backing_indices[1].key_count())
        builder.add_node(*nodes[4])
        builder.add_node(*nodes[5])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(2, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        self.assertEqual(4, builder._backing_indices[1].key_count())
        builder.add_node(*nodes[6])
        builder.add_node(*nodes[7])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(None, builder._backing_indices[1])
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[8])
        builder.add_node(*nodes[9])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        self.assertEqual(None, builder._backing_indices[1])
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[10])
        builder.add_node(*nodes[11])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(4, builder._backing_indices[1].key_count())
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[12])
        self.assertEqual([(builder,) + node for node in sorted(nodes[:13])], list(builder.iter_all_entries()))
        self.assertEqual(set([(builder,) + node for node in nodes[11:13]]), set(builder.iter_entries([nodes[12][0], nodes[11][0]])))
        self.assertEqual(13, builder.key_count())
        self.assertEqual(set([(builder,) + node for node in nodes[11:13]]), set(builder.iter_entries_prefix([nodes[12][0], nodes[11][0]])))
        builder.add_node(*nodes[13])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        self.assertEqual(4, builder._backing_indices[1].key_count())
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[14])
        builder.add_node(*nodes[15])
        self.assertEqual(4, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(None, builder._backing_indices[1])
        self.assertEqual(None, builder._backing_indices[2])
        self.assertEqual(16, builder._backing_indices[3].key_count())
        transport = self.get_transport('')
        size = transport.put_file('index', builder.finish())
        index = btree_index.BTreeGraphIndex(transport, 'index', size)
        nodes = list(index.iter_all_entries())
        self.assertEqual(sorted(nodes), nodes)
        self.assertEqual(16, len(nodes))

    def test_spill_index_duplicate_key_caught_on_finish(self):
        if False:
            while True:
                i = 10
        builder = btree_index.BTreeBuilder(key_elements=1, spill_at=2)
        nodes = [node[0:2] for node in self.make_nodes(16, 1, 0)]
        builder.add_node(*nodes[0])
        builder.add_node(*nodes[1])
        builder.add_node(*nodes[0])
        self.assertRaises(errors.BadIndexDuplicateKey, builder.finish)

class TestBTreeIndex(BTreeTestCase):

    def make_index(self, ref_lists=0, key_elements=1, nodes=[]):
        if False:
            return 10
        builder = btree_index.BTreeBuilder(reference_lists=ref_lists, key_elements=key_elements)
        for (key, value, references) in nodes:
            builder.add_node(key, value, references)
        stream = builder.finish()
        trans = transport.get_transport_from_url('trace+' + self.get_url())
        size = trans.put_file('index', stream)
        return btree_index.BTreeGraphIndex(trans, 'index', size)

    def make_index_with_offset(self, ref_lists=1, key_elements=1, nodes=[], offset=0):
        if False:
            while True:
                i = 10
        builder = btree_index.BTreeBuilder(key_elements=key_elements, reference_lists=ref_lists)
        builder.add_nodes(nodes)
        transport = self.get_transport('')
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        size = len(content)
        transport.put_bytes('index', ' ' * offset + content)
        return btree_index.BTreeGraphIndex(transport, 'index', size=size, offset=offset)

    def test_clear_cache(self):
        if False:
            while True:
                i = 10
        nodes = self.make_nodes(160, 2, 2)
        index = self.make_index(ref_lists=2, key_elements=2, nodes=nodes)
        self.assertEqual(1, len(list(index.iter_entries([nodes[30][0]]))))
        self.assertEqual([1, 4], index._row_lengths)
        self.assertIsNot(None, index._root_node)
        internal_node_pre_clear = index._internal_node_cache.keys()
        self.assertTrue(len(index._leaf_node_cache) > 0)
        index.clear_cache()
        self.assertIsNot(None, index._root_node)
        self.assertEqual(internal_node_pre_clear, index._internal_node_cache.keys())
        self.assertEqual(0, len(index._leaf_node_cache))

    def test_trivial_constructor(self):
        if False:
            return 10
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        index = btree_index.BTreeGraphIndex(t, 'index', None)
        self.assertEqual([], t._activity)

    def test_with_size_constructor(self):
        if False:
            for i in range(10):
                print('nop')
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        index = btree_index.BTreeGraphIndex(t, 'index', 1)
        self.assertEqual([], t._activity)

    def test_empty_key_count_no_size(self):
        if False:
            i = 10
            return i + 15
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        t.put_file('index', builder.finish())
        index = btree_index.BTreeGraphIndex(t, 'index', None)
        del t._activity[:]
        self.assertEqual([], t._activity)
        self.assertEqual(0, index.key_count())
        self.assertEqual([('get', 'index')], t._activity)

    def test_empty_key_count(self):
        if False:
            for i in range(10):
                print('nop')
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        self.assertEqual(72, size)
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        self.assertEqual(0, index.key_count())
        self.assertEqual([('readv', 'index', [(0, 72)], False, None)], t._activity)

    def test_non_empty_key_count_2_2(self):
        if False:
            while True:
                i = 10
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(35, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        self.assertEqual(70, index.key_count())
        self.assertEqual([('readv', 'index', [(0, size)], False, None)], t._activity)
        self.assertEqualApproxCompressed(1173, size)

    def test_with_offset_no_size(self):
        if False:
            i = 10
            return i + 15
        index = self.make_index_with_offset(key_elements=1, ref_lists=1, offset=1234, nodes=self.make_nodes(200, 1, 1))
        index._size = None
        self.assertEqual(200, index.key_count())

    def test_with_small_offset(self):
        if False:
            while True:
                i = 10
        index = self.make_index_with_offset(key_elements=1, ref_lists=1, offset=1234, nodes=self.make_nodes(200, 1, 1))
        self.assertEqual(200, index.key_count())

    def test_with_large_offset(self):
        if False:
            while True:
                i = 10
        index = self.make_index_with_offset(key_elements=1, ref_lists=1, offset=123456, nodes=self.make_nodes(200, 1, 1))
        self.assertEqual(200, index.key_count())

    def test__read_nodes_no_size_one_page_reads_once(self):
        if False:
            for i in range(10):
                print('nop')
        self.make_index(nodes=[(('key',), 'value', ())])
        trans = transport.get_transport_from_url('trace+' + self.get_url())
        index = btree_index.BTreeGraphIndex(trans, 'index', None)
        del trans._activity[:]
        nodes = dict(index._read_nodes([0]))
        self.assertEqual([0], nodes.keys())
        node = nodes[0]
        self.assertEqual([('key',)], node.all_keys())
        self.assertEqual([('get', 'index')], trans._activity)

    def test__read_nodes_no_size_multiple_pages(self):
        if False:
            i = 10
            return i + 15
        index = self.make_index(2, 2, nodes=self.make_nodes(160, 2, 2))
        index.key_count()
        num_pages = index._row_offsets[-1]
        trans = transport.get_transport_from_url('trace+' + self.get_url())
        index = btree_index.BTreeGraphIndex(trans, 'index', None)
        del trans._activity[:]
        nodes = dict(index._read_nodes([0]))
        self.assertEqual(range(num_pages), nodes.keys())

    def test_2_levels_key_count_2_2(self):
        if False:
            while True:
                i = 10
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(160, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        self.assertEqualApproxCompressed(17692, size)
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        self.assertEqual(320, index.key_count())
        self.assertEqual([('readv', 'index', [(0, 4096)], False, None)], t._activity)

    def test_validate_one_page(self):
        if False:
            print('Hello World!')
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(45, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        index.validate()
        self.assertEqual([('readv', 'index', [(0, size)], False, None)], t._activity)
        self.assertEqualApproxCompressed(1488, size)

    def test_validate_two_pages(self):
        if False:
            print('Hello World!')
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(80, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        self.assertEqualApproxCompressed(9339, size)
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        index.validate()
        rem = size - 8192
        self.assertEqual([('readv', 'index', [(0, 4096)], False, None), ('readv', 'index', [(4096, 4096), (8192, rem)], False, None)], t._activity)

    def test_eq_ne(self):
        if False:
            print('Hello World!')
        t1 = transport.get_transport_from_url('trace+' + self.get_url(''))
        t2 = self.get_transport()
        self.assertTrue(btree_index.BTreeGraphIndex(t1, 'index', None) == btree_index.BTreeGraphIndex(t1, 'index', None))
        self.assertTrue(btree_index.BTreeGraphIndex(t1, 'index', 20) == btree_index.BTreeGraphIndex(t1, 'index', 20))
        self.assertFalse(btree_index.BTreeGraphIndex(t1, 'index', 20) == btree_index.BTreeGraphIndex(t2, 'index', 20))
        self.assertFalse(btree_index.BTreeGraphIndex(t1, 'inde1', 20) == btree_index.BTreeGraphIndex(t1, 'inde2', 20))
        self.assertFalse(btree_index.BTreeGraphIndex(t1, 'index', 10) == btree_index.BTreeGraphIndex(t1, 'index', 20))
        self.assertFalse(btree_index.BTreeGraphIndex(t1, 'index', None) != btree_index.BTreeGraphIndex(t1, 'index', None))
        self.assertFalse(btree_index.BTreeGraphIndex(t1, 'index', 20) != btree_index.BTreeGraphIndex(t1, 'index', 20))
        self.assertTrue(btree_index.BTreeGraphIndex(t1, 'index', 20) != btree_index.BTreeGraphIndex(t2, 'index', 20))
        self.assertTrue(btree_index.BTreeGraphIndex(t1, 'inde1', 20) != btree_index.BTreeGraphIndex(t1, 'inde2', 20))
        self.assertTrue(btree_index.BTreeGraphIndex(t1, 'index', 10) != btree_index.BTreeGraphIndex(t1, 'index', 20))

    def test_key_too_big(self):
        if False:
            while True:
                i = 10
        bigKey = ''.join(map(repr, xrange(btree_index._PAGE_SIZE)))
        self.assertRaises(errors.BadIndexKey, self.make_index, nodes=[((bigKey,), 'value', ())])

    def test_iter_all_only_root_no_size(self):
        if False:
            i = 10
            return i + 15
        self.make_index(nodes=[(('key',), 'value', ())])
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        index = btree_index.BTreeGraphIndex(t, 'index', None)
        del t._activity[:]
        self.assertEqual([(('key',), 'value')], [x[1:] for x in index.iter_all_entries()])
        self.assertEqual([('get', 'index')], t._activity)

    def test_iter_all_entries_reads(self):
        if False:
            for i in range(10):
                print('nop')
        self.shrink_page_size()
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(10000, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        page_size = btree_index._PAGE_SIZE
        del builder
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        found_nodes = self.time(list, index.iter_all_entries())
        bare_nodes = []
        for node in found_nodes:
            self.assertTrue(node[0] is index)
            bare_nodes.append(node[1:])
        self.assertEqual(3, len(index._row_lengths), 'Not enough rows: %r' % index._row_lengths)
        self.assertEqual(20000, len(found_nodes))
        self.assertEqual(set(nodes), set(bare_nodes))
        total_pages = sum(index._row_lengths)
        self.assertEqual(total_pages, index._row_offsets[-1])
        self.assertEqualApproxCompressed(1303220, size)
        first_byte = index._row_offsets[-2] * page_size
        readv_request = []
        for offset in range(first_byte, size, page_size):
            readv_request.append((offset, page_size))
        readv_request[-1] = (readv_request[-1][0], size % page_size)
        expected = [('readv', 'index', [(0, page_size)], False, None), ('readv', 'index', readv_request, False, None)]
        if expected != t._activity:
            self.assertEqualDiff(pprint.pformat(expected), pprint.pformat(t._activity))

    def _test_iter_entries_references_resolved(self):
        if False:
            return 10
        index = self.make_index(1, nodes=[(('name',), 'data', ([('ref',), ('ref',)],)), (('ref',), 'refdata', ([],))])
        self.assertEqual(set([(index, ('name',), 'data', ((('ref',), ('ref',)),)), (index, ('ref',), 'refdata', ((),))]), set(index.iter_entries([('name',), ('ref',)])))

    def test_iter_entries_references_2_refs_resolved(self):
        if False:
            print('Hello World!')
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(160, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        del builder
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        found_nodes = list(index.iter_entries([nodes[30][0]]))
        bare_nodes = []
        for node in found_nodes:
            self.assertTrue(node[0] is index)
            bare_nodes.append(node[1:])
        self.assertEqual(1, len(found_nodes))
        self.assertEqual(nodes[30], bare_nodes[0])
        self.assertEqual([('readv', 'index', [(0, 4096)], False, None), ('readv', 'index', [(8192, 4096)], False, None)], t._activity)

    def test_iter_key_prefix_1_element_key_None(self):
        if False:
            while True:
                i = 10
        index = self.make_index()
        self.assertRaises(errors.BadIndexKey, list, index.iter_entries_prefix([(None,)]))

    def test_iter_key_prefix_wrong_length(self):
        if False:
            while True:
                i = 10
        index = self.make_index()
        self.assertRaises(errors.BadIndexKey, list, index.iter_entries_prefix([('foo', None)]))
        index = self.make_index(key_elements=2)
        self.assertRaises(errors.BadIndexKey, list, index.iter_entries_prefix([('foo',)]))
        self.assertRaises(errors.BadIndexKey, list, index.iter_entries_prefix([('foo', None, None)]))

    def test_iter_key_prefix_1_key_element_no_refs(self):
        if False:
            i = 10
            return i + 15
        index = self.make_index(nodes=[(('name',), 'data', ()), (('ref',), 'refdata', ())])
        self.assertEqual(set([(index, ('name',), 'data'), (index, ('ref',), 'refdata')]), set(index.iter_entries_prefix([('name',), ('ref',)])))

    def test_iter_key_prefix_1_key_element_refs(self):
        if False:
            return 10
        index = self.make_index(1, nodes=[(('name',), 'data', ([('ref',)],)), (('ref',), 'refdata', ([],))])
        self.assertEqual(set([(index, ('name',), 'data', ((('ref',),),)), (index, ('ref',), 'refdata', ((),))]), set(index.iter_entries_prefix([('name',), ('ref',)])))

    def test_iter_key_prefix_2_key_element_no_refs(self):
        if False:
            while True:
                i = 10
        index = self.make_index(key_elements=2, nodes=[(('name', 'fin1'), 'data', ()), (('name', 'fin2'), 'beta', ()), (('ref', 'erence'), 'refdata', ())])
        self.assertEqual(set([(index, ('name', 'fin1'), 'data'), (index, ('ref', 'erence'), 'refdata')]), set(index.iter_entries_prefix([('name', 'fin1'), ('ref', 'erence')])))
        self.assertEqual(set([(index, ('name', 'fin1'), 'data'), (index, ('name', 'fin2'), 'beta')]), set(index.iter_entries_prefix([('name', None)])))

    def test_iter_key_prefix_2_key_element_refs(self):
        if False:
            while True:
                i = 10
        index = self.make_index(1, key_elements=2, nodes=[(('name', 'fin1'), 'data', ([('ref', 'erence')],)), (('name', 'fin2'), 'beta', ([],)), (('ref', 'erence'), 'refdata', ([],))])
        self.assertEqual(set([(index, ('name', 'fin1'), 'data', ((('ref', 'erence'),),)), (index, ('ref', 'erence'), 'refdata', ((),))]), set(index.iter_entries_prefix([('name', 'fin1'), ('ref', 'erence')])))
        self.assertEqual(set([(index, ('name', 'fin1'), 'data', ((('ref', 'erence'),),)), (index, ('name', 'fin2'), 'beta', ((),))]), set(index.iter_entries_prefix([('name', None)])))

    def test_external_references_no_refs(self):
        if False:
            for i in range(10):
                print('nop')
        index = self.make_index(ref_lists=0, nodes=[])
        self.assertRaises(ValueError, index.external_references, 0)

    def test_external_references_no_results(self):
        if False:
            for i in range(10):
                print('nop')
        index = self.make_index(ref_lists=1, nodes=[(('key',), 'value', ([],))])
        self.assertEqual(set(), index.external_references(0))

    def test_external_references_missing_ref(self):
        if False:
            return 10
        missing_key = ('missing',)
        index = self.make_index(ref_lists=1, nodes=[(('key',), 'value', ([missing_key],))])
        self.assertEqual(set([missing_key]), index.external_references(0))

    def test_external_references_multiple_ref_lists(self):
        if False:
            return 10
        missing_key = ('missing',)
        index = self.make_index(ref_lists=2, nodes=[(('key',), 'value', ([], [missing_key]))])
        self.assertEqual(set([]), index.external_references(0))
        self.assertEqual(set([missing_key]), index.external_references(1))

    def test_external_references_two_records(self):
        if False:
            print('Hello World!')
        index = self.make_index(ref_lists=1, nodes=[(('key-1',), 'value', ([('key-2',)],)), (('key-2',), 'value', ([],))])
        self.assertEqual(set([]), index.external_references(0))

    def test__find_ancestors_one_page(self):
        if False:
            i = 10
            return i + 15
        key1 = ('key-1',)
        key2 = ('key-2',)
        index = self.make_index(ref_lists=1, key_elements=1, nodes=[(key1, 'value', ([key2],)), (key2, 'value', ([],))])
        parent_map = {}
        missing_keys = set()
        search_keys = index._find_ancestors([key1], 0, parent_map, missing_keys)
        self.assertEqual({key1: (key2,), key2: ()}, parent_map)
        self.assertEqual(set(), missing_keys)
        self.assertEqual(set(), search_keys)

    def test__find_ancestors_one_page_w_missing(self):
        if False:
            i = 10
            return i + 15
        key1 = ('key-1',)
        key2 = ('key-2',)
        key3 = ('key-3',)
        index = self.make_index(ref_lists=1, key_elements=1, nodes=[(key1, 'value', ([key2],)), (key2, 'value', ([],))])
        parent_map = {}
        missing_keys = set()
        search_keys = index._find_ancestors([key2, key3], 0, parent_map, missing_keys)
        self.assertEqual({key2: ()}, parent_map)
        self.assertEqual(set([key3]), missing_keys)
        self.assertEqual(set(), search_keys)

    def test__find_ancestors_one_parent_missing(self):
        if False:
            for i in range(10):
                print('nop')
        key1 = ('key-1',)
        key2 = ('key-2',)
        key3 = ('key-3',)
        index = self.make_index(ref_lists=1, key_elements=1, nodes=[(key1, 'value', ([key2],)), (key2, 'value', ([key3],))])
        parent_map = {}
        missing_keys = set()
        search_keys = index._find_ancestors([key1], 0, parent_map, missing_keys)
        self.assertEqual({key1: (key2,), key2: (key3,)}, parent_map)
        self.assertEqual(set(), missing_keys)
        self.assertEqual(set([key3]), search_keys)
        search_keys = index._find_ancestors(search_keys, 0, parent_map, missing_keys)
        self.assertEqual({key1: (key2,), key2: (key3,)}, parent_map)
        self.assertEqual(set([key3]), missing_keys)
        self.assertEqual(set([]), search_keys)

    def test__find_ancestors_dont_search_known(self):
        if False:
            print('Hello World!')
        key1 = ('key-1',)
        key2 = ('key-2',)
        key3 = ('key-3',)
        index = self.make_index(ref_lists=1, key_elements=1, nodes=[(key1, 'value', ([key2],)), (key2, 'value', ([key3],)), (key3, 'value', ([],))])
        parent_map = {key2: (key3,)}
        missing_keys = set()
        search_keys = index._find_ancestors([key1], 0, parent_map, missing_keys)
        self.assertEqual({key1: (key2,), key2: (key3,)}, parent_map)
        self.assertEqual(set(), missing_keys)
        self.assertEqual(set(), search_keys)

    def test__find_ancestors_multiple_pages(self):
        if False:
            for i in range(10):
                print('nop')
        start_time = 1249671539
        email = 'joebob@example.com'
        nodes = []
        ref_lists = ((),)
        rev_keys = []
        for i in xrange(400):
            rev_id = '%s-%s-%s' % (email, osutils.compact_date(start_time + i), osutils.rand_chars(16))
            rev_key = (rev_id,)
            nodes.append((rev_key, 'value', ref_lists))
            ref_lists = ((rev_key,),)
            rev_keys.append(rev_key)
        index = self.make_index(ref_lists=1, key_elements=1, nodes=nodes)
        self.assertEqual(400, index.key_count())
        self.assertEqual(3, len(index._row_offsets))
        nodes = dict(index._read_nodes([1, 2]))
        l1 = nodes[1]
        l2 = nodes[2]
        min_l2_key = l2.min_key
        max_l1_key = l1.max_key
        self.assertTrue(max_l1_key < min_l2_key)
        parents_min_l2_key = l2[min_l2_key][1][0]
        self.assertEqual((l1.max_key,), parents_min_l2_key)
        key_idx = rev_keys.index(min_l2_key)
        next_key = rev_keys[key_idx + 1]
        parent_map = {}
        missing_keys = set()
        search_keys = index._find_ancestors([next_key], 0, parent_map, missing_keys)
        self.assertEqual([min_l2_key, next_key], sorted(parent_map))
        self.assertEqual(set(), missing_keys)
        self.assertEqual(set([max_l1_key]), search_keys)
        parent_map = {}
        search_keys = index._find_ancestors([max_l1_key], 0, parent_map, missing_keys)
        self.assertEqual(l1.all_keys(), sorted(parent_map))
        self.assertEqual(set(), missing_keys)
        self.assertEqual(set(), search_keys)

    def test__find_ancestors_empty_index(self):
        if False:
            print('Hello World!')
        index = self.make_index(ref_lists=1, key_elements=1, nodes=[])
        parent_map = {}
        missing_keys = set()
        search_keys = index._find_ancestors([('one',), ('two',)], 0, parent_map, missing_keys)
        self.assertEqual(set(), search_keys)
        self.assertEqual({}, parent_map)
        self.assertEqual(set([('one',), ('two',)]), missing_keys)

    def test_supports_unlimited_cache(self):
        if False:
            for i in range(10):
                print('nop')
        builder = btree_index.BTreeBuilder(reference_lists=0, key_elements=1)
        nodes = self.make_nodes(500, 1, 0)
        for node in nodes:
            builder.add_node(*node)
        stream = builder.finish()
        trans = self.get_transport()
        size = trans.put_file('index', stream)
        index = btree_index.BTreeGraphIndex(trans, 'index', size)
        self.assertEqual(500, index.key_count())
        self.assertEqual(2, len(index._row_lengths))
        self.assertTrue(index._row_lengths[-1] >= 2)
        self.assertIsInstance(index._leaf_node_cache, lru_cache.LRUCache)
        self.assertEqual(btree_index._NODE_CACHE_SIZE, index._leaf_node_cache._max_cache)
        self.assertIsInstance(index._internal_node_cache, fifo_cache.FIFOCache)
        self.assertEqual(100, index._internal_node_cache._max_cache)
        index = btree_index.BTreeGraphIndex(trans, 'index', size, unlimited_cache=False)
        self.assertIsInstance(index._leaf_node_cache, lru_cache.LRUCache)
        self.assertEqual(btree_index._NODE_CACHE_SIZE, index._leaf_node_cache._max_cache)
        self.assertIsInstance(index._internal_node_cache, fifo_cache.FIFOCache)
        self.assertEqual(100, index._internal_node_cache._max_cache)
        index = btree_index.BTreeGraphIndex(trans, 'index', size, unlimited_cache=True)
        self.assertIsInstance(index._leaf_node_cache, dict)
        self.assertIs(type(index._internal_node_cache), dict)
        entries = set(index.iter_entries([n[0] for n in nodes]))
        self.assertEqual(500, len(entries))

class TestBTreeNodes(BTreeTestCase):
    scenarios = btreeparser_scenarios()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestBTreeNodes, self).setUp()
        self.overrideAttr(btree_index, '_btree_serializer', self.parse_btree)

    def test_LeafNode_1_0(self):
        if False:
            i = 10
            return i + 15
        node_bytes = 'type=leaf\n0000000000000000000000000000000000000000\x00\x00value:0\n1111111111111111111111111111111111111111\x00\x00value:1\n2222222222222222222222222222222222222222\x00\x00value:2\n3333333333333333333333333333333333333333\x00\x00value:3\n4444444444444444444444444444444444444444\x00\x00value:4\n'
        node = btree_index._LeafNode(node_bytes, 1, 0)
        self.assertEqual({('0000000000000000000000000000000000000000',): ('value:0', ()), ('1111111111111111111111111111111111111111',): ('value:1', ()), ('2222222222222222222222222222222222222222',): ('value:2', ()), ('3333333333333333333333333333333333333333',): ('value:3', ()), ('4444444444444444444444444444444444444444',): ('value:4', ())}, dict(node.all_items()))

    def test_LeafNode_2_2(self):
        if False:
            for i in range(10):
                print('nop')
        node_bytes = 'type=leaf\n00\x0000\x00\t00\x00ref00\x00value:0\n00\x0011\x0000\x00ref00\t00\x00ref00\r01\x00ref01\x00value:1\n11\x0033\x0011\x00ref22\t11\x00ref22\r11\x00ref22\x00value:3\n11\x0044\x00\t11\x00ref00\x00value:4\n'
        node = btree_index._LeafNode(node_bytes, 2, 2)
        self.assertEqual({('00', '00'): ('value:0', ((), (('00', 'ref00'),))), ('00', '11'): ('value:1', ((('00', 'ref00'),), (('00', 'ref00'), ('01', 'ref01')))), ('11', '33'): ('value:3', ((('11', 'ref22'),), (('11', 'ref22'), ('11', 'ref22')))), ('11', '44'): ('value:4', ((), (('11', 'ref00'),)))}, dict(node.all_items()))

    def test_InternalNode_1(self):
        if False:
            while True:
                i = 10
        node_bytes = 'type=internal\noffset=1\n0000000000000000000000000000000000000000\n1111111111111111111111111111111111111111\n2222222222222222222222222222222222222222\n3333333333333333333333333333333333333333\n4444444444444444444444444444444444444444\n'
        node = btree_index._InternalNode(node_bytes)
        self.assertEqual([('0000000000000000000000000000000000000000',), ('1111111111111111111111111111111111111111',), ('2222222222222222222222222222222222222222',), ('3333333333333333333333333333333333333333',), ('4444444444444444444444444444444444444444',)], node.keys)
        self.assertEqual(1, node.offset)

    def test_LeafNode_2_2(self):
        if False:
            i = 10
            return i + 15
        node_bytes = 'type=leaf\n00\x0000\x00\t00\x00ref00\x00value:0\n00\x0011\x0000\x00ref00\t00\x00ref00\r01\x00ref01\x00value:1\n11\x0033\x0011\x00ref22\t11\x00ref22\r11\x00ref22\x00value:3\n11\x0044\x00\t11\x00ref00\x00value:4\n'
        node = btree_index._LeafNode(node_bytes, 2, 2)
        self.assertEqual({('00', '00'): ('value:0', ((), (('00', 'ref00'),))), ('00', '11'): ('value:1', ((('00', 'ref00'),), (('00', 'ref00'), ('01', 'ref01')))), ('11', '33'): ('value:3', ((('11', 'ref22'),), (('11', 'ref22'), ('11', 'ref22')))), ('11', '44'): ('value:4', ((), (('11', 'ref00'),)))}, dict(node.all_items()))

    def assertFlattened(self, expected, key, value, refs):
        if False:
            return 10
        (flat_key, flat_line) = self.parse_btree._flatten_node((None, key, value, refs), bool(refs))
        self.assertEqual('\x00'.join(key), flat_key)
        self.assertEqual(expected, flat_line)

    def test__flatten_node(self):
        if False:
            return 10
        self.assertFlattened('key\x00\x00value\n', ('key',), 'value', [])
        self.assertFlattened('key\x00tuple\x00\x00value str\n', ('key', 'tuple'), 'value str', [])
        self.assertFlattened('key\x00tuple\x00triple\x00\x00value str\n', ('key', 'tuple', 'triple'), 'value str', [])
        self.assertFlattened('k\x00t\x00s\x00ref\x00value str\n', ('k', 't', 's'), 'value str', [[('ref',)]])
        self.assertFlattened('key\x00tuple\x00ref\x00key\x00value str\n', ('key', 'tuple'), 'value str', [[('ref', 'key')]])
        self.assertFlattened('00\x0000\x00\t00\x00ref00\x00value:0\n', ('00', '00'), 'value:0', ((), (('00', 'ref00'),)))
        self.assertFlattened('00\x0011\x0000\x00ref00\t00\x00ref00\r01\x00ref01\x00value:1\n', ('00', '11'), 'value:1', ((('00', 'ref00'),), (('00', 'ref00'), ('01', 'ref01'))))
        self.assertFlattened('11\x0033\x0011\x00ref22\t11\x00ref22\r11\x00ref22\x00value:3\n', ('11', '33'), 'value:3', ((('11', 'ref22'),), (('11', 'ref22'), ('11', 'ref22'))))
        self.assertFlattened('11\x0044\x00\t11\x00ref00\x00value:4\n', ('11', '44'), 'value:4', ((), (('11', 'ref00'),)))

class TestCompiledBtree(tests.TestCase):

    def test_exists(self):
        if False:
            while True:
                i = 10
        self.requireFeature(compiled_btreeparser_feature)

class TestMultiBisectRight(tests.TestCase):

    def assertMultiBisectRight(self, offsets, search_keys, fixed_keys):
        if False:
            print('Hello World!')
        self.assertEqual(offsets, btree_index.BTreeGraphIndex._multi_bisect_right(search_keys, fixed_keys))

    def test_after(self):
        if False:
            while True:
                i = 10
        self.assertMultiBisectRight([(1, ['b'])], ['b'], ['a'])
        self.assertMultiBisectRight([(3, ['e', 'f', 'g'])], ['e', 'f', 'g'], ['a', 'b', 'c'])

    def test_before(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMultiBisectRight([(0, ['a'])], ['a'], ['b'])
        self.assertMultiBisectRight([(0, ['a', 'b', 'c', 'd'])], ['a', 'b', 'c', 'd'], ['e', 'f', 'g'])

    def test_exact(self):
        if False:
            while True:
                i = 10
        self.assertMultiBisectRight([(1, ['a'])], ['a'], ['a'])
        self.assertMultiBisectRight([(1, ['a']), (2, ['b'])], ['a', 'b'], ['a', 'b'])
        self.assertMultiBisectRight([(1, ['a']), (3, ['c'])], ['a', 'c'], ['a', 'b', 'c'])

    def test_inbetween(self):
        if False:
            while True:
                i = 10
        self.assertMultiBisectRight([(1, ['b'])], ['b'], ['a', 'c'])
        self.assertMultiBisectRight([(1, ['b', 'c', 'd']), (2, ['f', 'g'])], ['b', 'c', 'd', 'f', 'g'], ['a', 'e', 'h'])

    def test_mixed(self):
        if False:
            i = 10
            return i + 15
        self.assertMultiBisectRight([(0, ['a', 'b']), (2, ['d', 'e']), (4, ['g', 'h'])], ['a', 'b', 'd', 'e', 'g', 'h'], ['c', 'd', 'f', 'g'])

class TestExpandOffsets(tests.TestCase):

    def make_index(self, size, recommended_pages=None):
        if False:
            while True:
                i = 10
        "Make an index with a generic size.\n\n        This doesn't actually create anything on disk, it just primes a\n        BTreeGraphIndex with the recommended information.\n        "
        index = btree_index.BTreeGraphIndex(transport.get_transport_from_url('memory:///'), 'test-index', size=size)
        if recommended_pages is not None:
            index._recommended_pages = recommended_pages
        return index

    def set_cached_offsets(self, index, cached_offsets):
        if False:
            print('Hello World!')
        'Monkeypatch to give a canned answer for _get_offsets_for...().'

        def _get_offsets_to_cached_pages():
            if False:
                for i in range(10):
                    print('nop')
            cached = set(cached_offsets)
            return cached
        index._get_offsets_to_cached_pages = _get_offsets_to_cached_pages

    def prepare_index(self, index, node_ref_lists, key_length, key_count, row_lengths, cached_offsets):
        if False:
            while True:
                i = 10
        'Setup the BTreeGraphIndex with some pre-canned information.'
        index.node_ref_lists = node_ref_lists
        index._key_length = key_length
        index._key_count = key_count
        index._row_lengths = row_lengths
        index._compute_row_offsets()
        index._root_node = btree_index._InternalNode('internal\noffset=0\n')
        self.set_cached_offsets(index, cached_offsets)

    def make_100_node_index(self):
        if False:
            return 10
        index = self.make_index(4096 * 100, 6)
        self.prepare_index(index, node_ref_lists=0, key_length=1, key_count=1000, row_lengths=[1, 99], cached_offsets=[0, 50])
        return index

    def make_1000_node_index(self):
        if False:
            print('Hello World!')
        index = self.make_index(4096 * 1000, 6)
        self.prepare_index(index, node_ref_lists=0, key_length=1, key_count=90000, row_lengths=[1, 9, 990], cached_offsets=[0, 5, 500])
        return index

    def assertNumPages(self, expected_pages, index, size):
        if False:
            i = 10
            return i + 15
        index._size = size
        self.assertEqual(expected_pages, index._compute_total_pages_in_index())

    def assertExpandOffsets(self, expected, index, offsets):
        if False:
            while True:
                i = 10
        self.assertEqual(expected, index._expand_offsets(offsets), 'We did not get the expected value after expanding %s' % (offsets,))

    def test_default_recommended_pages(self):
        if False:
            i = 10
            return i + 15
        index = self.make_index(None)
        self.assertEqual(1, index._recommended_pages)

    def test__compute_total_pages_in_index(self):
        if False:
            return 10
        index = self.make_index(None)
        self.assertNumPages(1, index, 1024)
        self.assertNumPages(1, index, 4095)
        self.assertNumPages(1, index, 4096)
        self.assertNumPages(2, index, 4097)
        self.assertNumPages(2, index, 8192)
        self.assertNumPages(76, index, 4096 * 75 + 10)

    def test__find_layer_start_and_stop(self):
        if False:
            print('Hello World!')
        index = self.make_1000_node_index()
        self.assertEqual((0, 1), index._find_layer_first_and_end(0))
        self.assertEqual((1, 10), index._find_layer_first_and_end(1))
        self.assertEqual((1, 10), index._find_layer_first_and_end(9))
        self.assertEqual((10, 1000), index._find_layer_first_and_end(10))
        self.assertEqual((10, 1000), index._find_layer_first_and_end(99))
        self.assertEqual((10, 1000), index._find_layer_first_and_end(999))

    def test_unknown_size(self):
        if False:
            print('Hello World!')
        index = self.make_index(None, 10)
        self.assertExpandOffsets([0], index, [0])
        self.assertExpandOffsets([1, 4, 9], index, [1, 4, 9])

    def test_more_than_recommended(self):
        if False:
            for i in range(10):
                print('nop')
        index = self.make_index(4096 * 100, 2)
        self.assertExpandOffsets([1, 10], index, [1, 10])
        self.assertExpandOffsets([1, 10, 20], index, [1, 10, 20])

    def test_read_all_from_root(self):
        if False:
            return 10
        index = self.make_index(4096 * 10, 20)
        self.assertExpandOffsets(range(10), index, [0])

    def test_read_all_when_cached(self):
        if False:
            print('Hello World!')
        index = self.make_index(4096 * 10, 5)
        self.prepare_index(index, node_ref_lists=0, key_length=1, key_count=1000, row_lengths=[1, 9], cached_offsets=[0, 1, 2, 5, 6])
        self.assertExpandOffsets([3, 4, 7, 8, 9], index, [3])
        self.assertExpandOffsets([3, 4, 7, 8, 9], index, [8])
        self.assertExpandOffsets([3, 4, 7, 8, 9], index, [9])

    def test_no_root_node(self):
        if False:
            i = 10
            return i + 15
        index = self.make_index(4096 * 10, 5)
        self.assertExpandOffsets([0], index, [0])

    def test_include_neighbors(self):
        if False:
            for i in range(10):
                print('nop')
        index = self.make_100_node_index()
        self.assertExpandOffsets([9, 10, 11, 12, 13, 14, 15], index, [12])
        self.assertExpandOffsets([88, 89, 90, 91, 92, 93, 94], index, [91])
        self.assertExpandOffsets([1, 2, 3, 4, 5, 6], index, [2])
        self.assertExpandOffsets([94, 95, 96, 97, 98, 99], index, [98])
        self.assertExpandOffsets([1, 2, 3, 80, 81, 82], index, [2, 81])
        self.assertExpandOffsets([1, 2, 3, 9, 10, 11, 80, 81, 82], index, [2, 10, 81])

    def test_stop_at_cached(self):
        if False:
            print('Hello World!')
        index = self.make_100_node_index()
        self.set_cached_offsets(index, [0, 10, 19])
        self.assertExpandOffsets([11, 12, 13, 14, 15, 16], index, [11])
        self.assertExpandOffsets([11, 12, 13, 14, 15, 16], index, [12])
        self.assertExpandOffsets([12, 13, 14, 15, 16, 17, 18], index, [15])
        self.assertExpandOffsets([13, 14, 15, 16, 17, 18], index, [16])
        self.assertExpandOffsets([13, 14, 15, 16, 17, 18], index, [17])
        self.assertExpandOffsets([13, 14, 15, 16, 17, 18], index, [18])

    def test_cannot_fully_expand(self):
        if False:
            return 10
        index = self.make_100_node_index()
        self.set_cached_offsets(index, [0, 10, 12])
        self.assertExpandOffsets([11], index, [11])

    def test_overlap(self):
        if False:
            return 10
        index = self.make_100_node_index()
        self.assertExpandOffsets([10, 11, 12, 13, 14, 15], index, [12, 13])
        self.assertExpandOffsets([10, 11, 12, 13, 14, 15], index, [11, 14])

    def test_stay_within_layer(self):
        if False:
            print('Hello World!')
        index = self.make_1000_node_index()
        self.assertExpandOffsets([1, 2, 3, 4], index, [2])
        self.assertExpandOffsets([6, 7, 8, 9], index, [6])
        self.assertExpandOffsets([6, 7, 8, 9], index, [9])
        self.assertExpandOffsets([10, 11, 12, 13, 14, 15], index, [10])
        self.assertExpandOffsets([10, 11, 12, 13, 14, 15, 16], index, [13])
        self.set_cached_offsets(index, [0, 4, 12])
        self.assertExpandOffsets([5, 6, 7, 8, 9], index, [7])
        self.assertExpandOffsets([10, 11], index, [11])

    def test_small_requests_unexpanded(self):
        if False:
            print('Hello World!')
        index = self.make_100_node_index()
        self.set_cached_offsets(index, [0])
        self.assertExpandOffsets([1], index, [1])
        self.assertExpandOffsets([50], index, [50])
        self.assertExpandOffsets([49, 50, 51, 59, 60, 61], index, [50, 60])
        index = self.make_1000_node_index()
        self.set_cached_offsets(index, [0])
        self.assertExpandOffsets([1], index, [1])
        self.set_cached_offsets(index, [0, 1])
        self.assertExpandOffsets([100], index, [100])
        self.set_cached_offsets(index, [0, 1, 100])
        self.assertExpandOffsets([2, 3, 4, 5, 6, 7], index, [2])
        self.assertExpandOffsets([2, 3, 4, 5, 6, 7], index, [4])
        self.set_cached_offsets(index, [0, 1, 2, 3, 4, 5, 6, 7, 100])
        self.assertExpandOffsets([102, 103, 104, 105, 106, 107, 108], index, [105])