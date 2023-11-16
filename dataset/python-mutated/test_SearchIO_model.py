"""Tests for SearchIO objects.

Tests the methods and behaviors of QueryResult, Hit, and HSP objects. All tests
are format-independent and are meant to check the fundamental behavior common
to all formats.

"""
import pickle
import unittest
from io import BytesIO
from copy import deepcopy
from search_tests_common import SearchTestBaseClass
from Bio.Align import MultipleSeqAlignment
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
frag111 = HSPFragment('hit1', 'query1', hit='ATGCGCAT', query='ATGCGCAT')
frag112 = HSPFragment('hit1', 'query1', hit='ATG', query='GAT')
frag113 = HSPFragment('hit1', 'query1', hit='ATTCG', query='AT-CG')
frag113b = HSPFragment('hit1', 'query1', hit='ATTCG', query='AT-CG')
frag114 = HSPFragment('hit1', 'query1', hit='AT', query='AT')
frag114b = HSPFragment('hit1', 'query1', hit='ATCG', query='ATGG')
frag211 = HSPFragment('hit2', 'query1', hit='GGGCCC', query='GGGCC-')
frag311 = HSPFragment('hit3', 'query1', hit='GATG', query='GTTG')
frag312 = HSPFragment('hit3', 'query1', hit='ATATAT', query='ATATAT')
frag411 = HSPFragment('hit4', 'query1', hit='CC-ATG', query='CCCATG')
frag121 = HSPFragment('hit1', 'query2', hit='GCGAG', query='GCGAC')
hsp111 = HSP([frag111])
hsp112 = HSP([frag112])
hsp113 = HSP([frag113, frag113b])
hsp114 = HSP([frag114, frag114b])
hsp211 = HSP([frag211])
hsp311 = HSP([frag311])
hsp312 = HSP([frag312])
hsp411 = HSP([frag411])
hsp121 = HSP([frag121])
hit11 = Hit([hsp111, hsp112, hsp113, hsp114])
hit21 = Hit([hsp211])
hit31 = Hit([hsp311, hsp312])
hit41 = Hit([hsp411])
hit12 = Hit([hsp121])

class QueryResultCases(SearchTestBaseClass):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.qresult = QueryResult([hit11, hit21, hit31], 'query1')
        self.qresult.seq_len = 1102
        self.qresult.target = 'refseq_rna'

    def test_pickle(self):
        if False:
            return 10
        'Test pickling and unpickling of QueryResult.'
        buf = BytesIO()
        pickle.dump(self.qresult, buf)
        unp = pickle.loads(buf.getvalue())
        self.compare_search_obj(self.qresult, unp)

    def test_order(self):
        if False:
            return 10
        self.assertEqual(self.qresult[0], hit11)
        self.assertEqual(self.qresult[2], hit31)
        del self.qresult['hit2']
        self.assertEqual(self.qresult[0], hit11)
        self.assertEqual(self.qresult[1], hit31)

    def test_init_none(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QueryResult.__init__, no arguments.'
        qresult = QueryResult()
        self.assertIsNone(qresult.id)
        self.assertIsNone(qresult.description)

    def test_init_id_only(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.__init__, with ID only.'
        qresult = QueryResult(id='query1')
        self.assertEqual('query1', qresult.id)
        self.assertIsNone(qresult.description)

    def test_init_hits_only(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.__init__, with hits only.'
        qresult = QueryResult([hit11, hit21, hit31])
        self.assertEqual('query1', qresult.id)
        self.assertEqual('<unknown description>', qresult.description)

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QueryResult.__repr__.'
        self.assertEqual("QueryResult(id='query1', 3 hits)", repr(self.qresult))

    def test_iter(self):
        if False:
            return 10
        'Test QueryResult.__iter__.'
        for (counter, hit) in enumerate(self.qresult):
            self.assertIn(hit, (hit11, hit21, hit31))
        self.assertEqual(2, counter)

    def test_hits(self):
        if False:
            print('Hello World!')
        'Test QueryResult.hits.'
        hits = list(self.qresult.hits)
        self.assertEqual([hit11, hit21, hit31], hits)

    def test_hit_keys(self):
        if False:
            return 10
        'Test QueryResult.hit_keys.'
        hit_keys = list(self.qresult.hit_keys)
        self.assertEqual(['hit1', 'hit2', 'hit3'], hit_keys)

    def test_items(self):
        if False:
            print('Hello World!')
        'Test QueryResult.items.'
        items = list(self.qresult.items)
        self.assertEqual([('hit1', hit11), ('hit2', hit21), ('hit3', hit31)], items)

    def test_hsps(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.hsps.'
        hsps = self.qresult.hsps
        self.assertEqual([hsp111, hsp112, hsp113, hsp114, hsp211, hsp311, hsp312], hsps)

    def test_fragments(self):
        if False:
            print('Hello World!')
        'Test QueryResult.fragments.'
        frags = self.qresult.fragments
        self.assertEqual([frag111, frag112, frag113, frag113b, frag114, frag114b, frag211, frag311, frag312], frags)

    def test_contains(self):
        if False:
            print('Hello World!')
        'Test QueryResult.__contains__.'
        self.assertIn('hit1', self.qresult)
        self.assertIn(hit21, self.qresult)
        self.assertNotIn('hit5', self.qresult)
        self.assertNotIn(hit41, self.qresult)

    def test_contains_alt(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.__contains__, with alternative IDs.'
        hit11._id_alt = ['alt1']
        query = QueryResult([hit11])
        self.assertIn('alt1', query)
        hit11._id_alt = []

    def test_len(self):
        if False:
            return 10
        'Test QueryResult.__len__.'
        self.assertEqual(3, len(self.qresult))

    def test_bool(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QueryResult.__bool__.'
        self.assertTrue(self.qresult)
        blank_qresult = QueryResult()
        self.assertFalse(blank_qresult)

    def test_setitem_ok(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QueryResult.__setitem__.'
        self.qresult['hit4'] = hit41
        self.assertEqual([hit11, hit21, hit31, hit41], list(self.qresult.hits))
        self.qresult['hit4'] = hit11
        self.assertEqual([hit11, hit21, hit31, hit11], list(self.qresult.hits))

    def test_setitem_ok_alt(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.__setitem__, checking alt hit IDs.'
        hit11._id_alt = ['alt1', 'alt11']
        query = QueryResult()
        query['hit1'] = hit11
        self.assertEqual(hit11, query['hit1'])
        self.assertEqual(hit11, query['alt1'])
        self.assertEqual(hit11, query['alt11'])
        self.assertNotEqual(hit11.id, 'alt1')
        self.assertNotEqual(hit11.id, 'alt11')
        hit11._id_alt = []

    def test_setitem_ok_alt_existing(self):
        if False:
            print('Hello World!')
        'Test QueryResult.__setitem__, existing key.'
        hit11._id_alt = ['alt1']
        hit21._id_alt = ['alt2']
        query = QueryResult()
        query['hit'] = hit11
        self.assertEqual(hit11, query['hit'])
        self.assertEqual(hit11, query['alt1'])
        query['hit'] = hit21
        self.assertEqual(hit21, query['hit'])
        self.assertEqual(hit21, query['alt2'])
        self.assertRaises(KeyError, query.__getitem__, 'alt1')
        hit11._id_alt = []
        hit21._id_alt = []

    def test_setitem_ok_alt_ok_promote(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QueryResult.__setitem__, previously alt ID.'
        hit11._id_alt = ['alt1']
        hit41._id_alt = ['alt4']
        hit31._id_alt = ['alt3']
        query = QueryResult([hit11, hit41])
        self.assertEqual(hit11, query['alt1'])
        self.assertEqual(hit41, query['alt4'])
        self.assertNotIn('alt1', query._items)
        self.assertIn('alt1', query._QueryResult__alt_hit_ids)
        query['alt1'] = hit31
        self.assertEqual(hit31, query['alt1'])
        self.assertEqual(hit41, query['alt4'])
        self.assertIn('alt1', query._items)
        self.assertNotIn('alt1', query._QueryResult__alt_hit_ids)
        hit11._id_alt = []
        hit41._id_alt = []
        hit31._id_alt = []

    def test_setitem_wrong_key_type(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.__setitem__, wrong key type.'
        self.assertRaises(TypeError, self.qresult.__setitem__, 0, hit41)
        self.assertRaises(TypeError, self.qresult.__setitem__, slice(0, 2), [hit41, hit31])

    def test_setitem_wrong_type(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.__setitem__, wrong type.'
        self.assertRaises(TypeError, self.qresult.__setitem__, 'hit4', hsp111)
        self.assertRaises(TypeError, self.qresult.__setitem__, 'hit5', 'hit5')

    def test_setitem_wrong_query_id(self):
        if False:
            return 10
        'Test QueryResult.__setitem__, wrong query ID.'
        self.assertRaises(ValueError, self.qresult.__setitem__, 'hit4', hit12)

    def test_setitem_from_empty(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.__setitem__, from empty container.'
        qresult = QueryResult()
        self.assertIsNone(qresult.id)
        self.assertIsNone(qresult.description)
        qresult.append(hit11)
        self.assertEqual('query1', qresult.id)
        self.assertEqual('<unknown description>', qresult.description)
        qresult.pop()
        self.assertEqual('query1', qresult.id)
        self.assertEqual('<unknown description>', qresult.description)

    def test_getitem_default_ok(self):
        if False:
            print('Hello World!')
        'Test QueryResult.__getitem__.'
        self.assertEqual(hit21, self.qresult['hit2'])
        self.assertEqual(hit11, self.qresult['hit1'])

    def test_getitem_int_ok(self):
        if False:
            return 10
        'Test QueryResult.__getitem__, with integer.'
        self.assertEqual(hit21, self.qresult[1])
        self.assertEqual(hit31, self.qresult[-1])

    def test_getitem_slice_ok(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.__getitem__, with slice.'
        self.assertEqual(1102, self.qresult.seq_len)
        self.assertEqual('refseq_rna', self.qresult.target)
        new_qresult = self.qresult[1:]
        self.assertEqual([hit21, hit31], list(new_qresult.hits))
        self.assertEqual(1102, new_qresult.seq_len)
        self.assertEqual('refseq_rna', new_qresult.target)

    def test_getitm_slice_alt_ok(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.__getitem__, with slice and alt IDs.'
        hit31._id_alt = ['alt3']
        hit11._id_alt = ['alt1']
        query = QueryResult([hit31, hit11])
        self.assertEqual(hit11, query['hit1'])
        self.assertEqual(hit11, query['alt1'])
        self.assertEqual(hit31, query['hit3'])
        self.assertEqual(hit31, query['alt3'])
        query = query[:1]
        self.assertEqual(hit31, query['hit3'])
        self.assertEqual(hit31, query['alt3'])
        self.assertRaises(KeyError, query.__getitem__, 'hit1')
        self.assertRaises(KeyError, query.__getitem__, 'alt1')
        hit31._id_alt = []
        hit11._id_alt = []

    def test_getitem_alt_ok(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.__getitem__, single item with alternative ID.'
        hit11._id_alt = ['alt1']
        query = QueryResult([hit11])
        self.assertEqual(hit11, query['hit1'])
        self.assertEqual(hit11, query['alt1'])
        self.assertNotEqual(hit11.id, 'alt1')
        hit11._id_alt = []

    def test_delitem_string_ok(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.__getitem__, with string.'
        del self.qresult['hit1']
        self.assertEqual(2, len(self.qresult))
        self.assertTrue([hit21, hit31], list(self.qresult.hits))

    def test_delitem_int_ok(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.__delitem__.'
        del self.qresult[-1]
        self.assertEqual(2, len(self.qresult))
        self.assertEqual([hit11, hit21], list(self.qresult.hits))
        del self.qresult[0]
        self.assertEqual(1, len(self.qresult))
        self.assertTrue([hit21], list(self.qresult.hits))

    def test_delitem_slice_ok(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.__delitem__, with slice.'
        del self.qresult[:-1]
        self.assertEqual(1, len(self.qresult))
        self.assertTrue([hit31], self.qresult.hits)

    def test_delitem_alt_ok(self):
        if False:
            return 10
        'Test QueryResult.__delitem__, with alt ID.'
        hit31._id_alt = ['alt3']
        qresult = QueryResult([hit31, hit41])
        self.assertEqual(2, len(qresult))
        del qresult['alt3']
        self.assertEqual(1, len(qresult))
        self.assertEqual(hit41, qresult['hit4'])
        self.assertRaises(KeyError, qresult.__getitem__, 'alt3')
        hit31._id_alt = []

    def test_description_set(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.description setter.'
        qresult = deepcopy(self.qresult)
        new_desc = 'unicorn hox homolog'
        for hit in qresult:
            self.assertNotEqual(new_desc, hit.query_description)
            for hsp in hit:
                self.assertNotEqual(new_desc, hsp.query_description)
                for fragment in hsp:
                    self.assertNotEqual(new_desc, fragment.query_description)
                    self.assertNotEqual(new_desc, fragment.query.description)
        qresult.description = new_desc
        for hit in qresult:
            self.assertEqual(new_desc, hit.query_description)
            for hsp in hit:
                self.assertEqual(new_desc, hsp.query_description)
                for fragment in hsp:
                    self.assertEqual(new_desc, fragment.query_description)
                    self.assertEqual(new_desc, fragment.query.description)

    def test_description_set_no_seqrecord(self):
        if False:
            return 10
        'Test QueryResult.description setter, without HSP SeqRecords.'
        frag1 = HSPFragment('hit1', 'query')
        frag2 = HSPFragment('hit1', 'query')
        frag3 = HSPFragment('hit2', 'query')
        hit1 = Hit([HSP([x]) for x in [frag1, frag2]])
        hit2 = Hit([HSP([frag3])])
        qresult = QueryResult([hit1, hit2])
        for hit in qresult:
            for hsp in hit.hsps:
                self.assertIsNone(getattr(hsp, 'query'))
        qresult.description = 'unicorn hox homolog'
        for hit in qresult:
            for hsp in hit.hsps:
                self.assertIsNone(getattr(hsp, 'query'))

    def test_id_set(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.id setter.'
        qresult = deepcopy(self.qresult)
        self.assertEqual('query1', qresult.id)
        for hit in qresult:
            self.assertEqual('query1', hit.query_id)
            for hsp in hit:
                self.assertEqual('query1', hsp.query_id)
                for fragment in hsp:
                    self.assertEqual('query1', fragment.query_id)
                    self.assertEqual('query1', fragment.query.id)
        qresult.id = 'new_id'
        self.assertEqual('new_id', qresult.id)
        for hit in qresult:
            self.assertEqual('new_id', hit.query_id)
            for hsp in hit:
                self.assertEqual('new_id', hsp.query_id)
                for fragment in hsp:
                    self.assertEqual('new_id', fragment.query_id)
                    self.assertEqual('new_id', fragment.query.id)

    def test_absorb_hit_does_not_exist(self):
        if False:
            print('Hello World!')
        'Test QueryResult.absorb, hit does not exist.'
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))
        self.qresult.absorb(hit41)
        self.assertEqual([hit11, hit21, hit31, hit41], list(self.qresult.hits))
        self.assertEqual(['hit1', 'hit2', 'hit3', 'hit4'], list(self.qresult.hit_keys))

    def test_absorb_hit_exists(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QueryResult.absorb, hit with the same ID exists.'
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))
        self.assertEqual(2, len(self.qresult['hit3']))
        hit = Hit([HSP([HSPFragment('hit3', 'query1')])])
        self.qresult.absorb(hit)
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))
        self.assertEqual(['hit1', 'hit2', 'hit3'], list(self.qresult.hit_keys))
        self.assertEqual(3, len(self.qresult['hit3']))
        del self.qresult['hit3'][-1]

    def test_append_ok(self):
        if False:
            return 10
        'Test QueryResult.append.'
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))
        self.qresult.append(hit41)
        self.assertEqual([hit11, hit21, hit31, hit41], list(self.qresult.hits))
        self.assertEqual(['hit1', 'hit2', 'hit3', 'hit4'], list(self.qresult.hit_keys))

    def test_append_custom_hit_key_function_ok(self):
        if False:
            print('Hello World!')
        'Test QueryResult.append, with custom hit key function.'
        self.qresult._hit_key_function = lambda hit: hit.id + '_custom'
        self.assertEqual(['hit1', 'hit2', 'hit3'], list(self.qresult.hit_keys))
        self.qresult.append(hit41)
        self.assertEqual(['hit1', 'hit2', 'hit3', 'hit4_custom'], list(self.qresult.hit_keys))

    def test_append_id_exists(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.append, when ID exists.'
        self.assertRaises(ValueError, self.qresult.append, hit11)

    def test_append_alt_id_exists(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.append, when alt ID exists.'
        hit11._id_alt = ['alt']
        hit21._id_alt = ['alt']
        qresult = QueryResult([hit11])
        self.assertRaises(ValueError, qresult.append, hit21)
        hit11._id_alt = []
        hit21._id_alt = []

    def test_append_alt_id_exists_alt(self):
        if False:
            return 10
        'Test QueryResult.append, when alt ID exists as primary.'
        hit21._id_alt = ['hit1']
        qresult = QueryResult([hit11])
        self.assertRaises(ValueError, qresult.append, hit21)
        hit21._id_alt = []

    def test_hit_filter(self):
        if False:
            print('Hello World!')
        'Test QueryResult.hit_filter.'
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))
        filter_func = lambda hit: len(hit) >= 2
        filtered = self.qresult.hit_filter(filter_func)
        self.assertEqual([hit11, hit31], list(filtered.hits))
        self.assertTrue(all((filter_func(hit) for hit in filtered)))
        self.assertEqual(1102, filtered.seq_len)
        self.assertEqual('refseq_rna', filtered.target)

    def test_hit_filter_no_func(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QueryResult.hit_filter, without arguments.'
        filtered = self.qresult.hit_filter()
        self.compare_search_obj(filtered, self.qresult)
        self.assertNotEqual(id(filtered), id(self.qresult))
        self.assertEqual(1102, filtered.seq_len)
        self.assertEqual('refseq_rna', filtered.target)

    def test_hit_filter_no_filtered(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.hit_filter, all hits filtered out.'
        filter_func = lambda hit: len(hit) > 50
        filtered = self.qresult.hit_filter(filter_func)
        self.assertEqual(0, len(filtered))
        self.assertIsInstance(filtered, QueryResult)
        self.assertEqual(1102, filtered.seq_len)
        self.assertEqual('refseq_rna', filtered.target)

    def test_hit_map(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QueryResult.hit_map.'
        qresult = deepcopy(self.qresult)

        def map_func(hit):
            if False:
                print('Hello World!')
            hit.id = hit.id.upper()
            return hit
        self.assertEqual('hit1', qresult[0].id)
        self.assertEqual('hit2', qresult[1].id)
        self.assertEqual('hit3', qresult[2].id)
        mapped = qresult.hit_map(map_func)
        self.assertEqual('HIT1', mapped[0].id)
        self.assertEqual('HIT2', mapped[1].id)
        self.assertEqual('HIT3', mapped[2].id)
        self.assertEqual(1102, mapped.seq_len)
        self.assertEqual('refseq_rna', mapped.target)

    def test_hit_map_no_func(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.hit_map, without arguments.'
        mapped = self.qresult.hit_map()
        self.compare_search_obj(mapped, self.qresult)
        self.assertNotEqual(id(mapped), id(self.qresult))
        self.assertEqual(1102, mapped.seq_len)
        self.assertEqual('refseq_rna', mapped.target)

    def test_hsp_filter(self):
        if False:
            print('Hello World!')
        'Test QueryResult.hsp_filter.'
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))
        filter_func = lambda hsp: '-' not in hsp.fragments[0].query
        filtered = self.qresult.hsp_filter(filter_func)
        self.assertIn('hit1', filtered)
        self.assertNotIn('hit2', filtered)
        self.assertIn('hit3', filtered)
        self.assertTrue(all((hsp in filtered['hit1'] for hsp in [hsp111, hsp112, hsp114])))
        self.assertTrue(all((hsp in filtered['hit3'] for hsp in [hsp311, hsp312])))

    def test_hsp_filter_no_func(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.hsp_filter, no arguments.'
        filtered = self.qresult.hsp_filter()
        self.compare_search_obj(filtered, self.qresult)
        self.assertNotEqual(id(filtered), id(self.qresult))
        self.assertEqual(1102, filtered.seq_len)
        self.assertEqual('refseq_rna', filtered.target)

    def test_hsp_filter_no_filtered(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QueryResult.hsp_filter, all hits filtered out.'
        filter_func = lambda hsp: len(hsp) > 50
        filtered = self.qresult.hsp_filter(filter_func)
        self.assertEqual(0, len(filtered))
        self.assertIsInstance(filtered, QueryResult)
        self.assertEqual(1102, filtered.seq_len)
        self.assertEqual('refseq_rna', filtered.target)

    def test_hsp_map(self):
        if False:
            print('Hello World!')
        'Test QueryResult.hsp_map.'
        qresult = deepcopy(self.qresult)
        for hit in qresult:
            for hsp in hit:
                setattr(hsp, 'mock', 13)

        def map_func(hsp):
            if False:
                while True:
                    i = 10
            mapped_frags = [x[1:] for x in hsp]
            return HSP(mapped_frags)
        mapped = qresult.hsp_map(map_func)
        for hit in mapped:
            for hsp in hit.hsps:
                self.assertFalse(hasattr(hsp, 'mock'))
        self.assertEqual('TGCGCAT', mapped['hit1'][0][0].hit.seq)
        self.assertEqual('TGCGCAT', mapped['hit1'][0][0].query.seq)
        self.assertEqual('TG', mapped['hit1'][1][0].hit.seq)
        self.assertEqual('AT', mapped['hit1'][1][0].query.seq)
        self.assertEqual('TTCG', mapped['hit1'][2][0].hit.seq)
        self.assertEqual('T-CG', mapped['hit1'][2][0].query.seq)
        self.assertEqual('TTCG', mapped['hit1'][2][1].hit.seq)
        self.assertEqual('T-CG', mapped['hit1'][2][1].query.seq)
        self.assertEqual('T', mapped['hit1'][3][0].hit.seq)
        self.assertEqual('T', mapped['hit1'][3][0].query.seq)
        self.assertEqual('TCG', mapped['hit1'][3][1].hit.seq)
        self.assertEqual('TGG', mapped['hit1'][3][1].query.seq)
        self.assertEqual('GGCCC', mapped['hit2'][0][0].hit.seq)
        self.assertEqual('GGCC-', mapped['hit2'][0][0].query.seq)
        self.assertEqual('ATG', mapped['hit3'][0][0].hit.seq)
        self.assertEqual('TTG', mapped['hit3'][0][0].query.seq)
        self.assertEqual('TATAT', mapped['hit3'][1][0].hit.seq)
        self.assertEqual('TATAT', mapped['hit3'][1][0].query.seq)
        self.assertEqual(1102, mapped.seq_len)
        self.assertEqual('refseq_rna', mapped.target)

    def test_hsp_map_no_func(self):
        if False:
            return 10
        'Test QueryResult.hsp_map, without arguments.'
        mapped = self.qresult.hsp_map()
        self.compare_search_obj(mapped, self.qresult)
        self.assertNotEqual(id(mapped), id(self.qresult))
        self.assertEqual(1102, mapped.seq_len)
        self.assertEqual('refseq_rna', mapped.target)

    def test_pop_nonexistent_with_default(self):
        if False:
            print('Hello World!')
        'Test QueryResult.pop with default for nonexistent key.'
        default = 'An arbitrary default return value for this test only.'
        nonexistent_key = 'neither a standard nor alternative key'
        hit = self.qresult.pop(nonexistent_key, default)
        self.assertEqual(hit, default)

    def test_pop_nonexistent_key(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QueryResult.pop with default for nonexistent key.'
        nonexistent_key = 'neither a standard nor alternative key'
        self.assertRaises(KeyError, self.qresult.pop, nonexistent_key)

    def test_pop_ok(self):
        if False:
            return 10
        'Test QueryResult.pop.'
        self.assertEqual(3, len(self.qresult))
        hit = self.qresult.pop()
        self.assertEqual(hit, hit31)
        self.assertEqual([hit11, hit21], list(self.qresult.hits))

    def test_pop_int_index_ok(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.pop, with integer index.'
        self.assertEqual(3, len(self.qresult))
        hit = self.qresult.pop(1)
        self.assertEqual(hit, hit21)
        self.assertEqual([hit11, hit31], list(self.qresult.hits))

    def test_pop_string_index_ok(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.pop, with string index.'
        self.assertEqual(3, len(self.qresult))
        hit = self.qresult.pop('hit2')
        self.assertEqual(hit, hit21)
        self.assertEqual([hit11, hit31], list(self.qresult.hits))

    def test_pop_string_alt_ok(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.pop, with alternative ID.'
        hit11._id_alt = ['alt1']
        hit21._id_alt = ['alt2']
        qresult = QueryResult([hit11, hit21])
        hit = qresult.pop('alt1')
        self.assertEqual(hit, hit11)
        self.assertEqual([hit21], list(qresult))
        self.assertNotIn('hit1', qresult)
        hit11._id_alt = []
        hit21._id_alt = []

    def test_index(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.index.'
        self.assertEqual(2, self.qresult.index('hit3'))
        self.assertEqual(2, self.qresult.index(hit31))

    def test_index_alt(self):
        if False:
            return 10
        'Test QueryResult.index, with alt ID.'
        hit11._id_alt = ['alt1']
        qresult = QueryResult([hit21, hit11])
        self.assertEqual(1, qresult.index('alt1'))
        hit11._id_alt = []

    def test_index_not_present(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.index, when index is not present.'
        self.assertRaises(ValueError, self.qresult.index, 'hit4')
        self.assertRaises(ValueError, self.qresult.index, hit41)

    def test_sort_ok(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QueryResult.sort.'
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))
        self.qresult.sort()
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))

    def test_sort_not_in_place_ok(self):
        if False:
            i = 10
            return i + 15
        'Test QueryResult.sort, not in place.'
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))
        sorted_qresult = self.qresult.sort(in_place=False)
        self.assertEqual([hit11, hit21, hit31], list(sorted_qresult.hits))
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))

    def test_sort_reverse_ok(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.sort, reverse.'
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))
        self.qresult.sort(reverse=True)
        self.assertEqual([hit31, hit21, hit11], list(self.qresult.hits))

    def test_sort_reverse_not_in_place_ok(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.sort, reverse, not in place.'
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))
        sorted_qresult = self.qresult.sort(reverse=True, in_place=False)
        self.assertEqual([hit31, hit21, hit11], list(sorted_qresult.hits))
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))

    def test_sort_key_ok(self):
        if False:
            for i in range(10):
                print('nop')
        'Test QueryResult.sort, with custom key.'
        key = lambda hit: len(hit)
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))
        self.qresult.sort(key=key)
        self.assertEqual([hit21, hit31, hit11], list(self.qresult.hits))

    def test_sort_key_not_in_place_ok(self):
        if False:
            while True:
                i = 10
        'Test QueryResult.sort, with custom key, not in place.'
        key = lambda hit: len(hit)
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))
        sorted_qresult = self.qresult.sort(key=key, in_place=False)
        self.assertEqual([hit21, hit31, hit11], list(sorted_qresult.hits))
        self.assertEqual([hit11, hit21, hit31], list(self.qresult.hits))

class HitCases(SearchTestBaseClass):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.hit = Hit([hsp111, hsp112, hsp113])
        self.hit.evalue = 5e-10
        self.hit.name = 'test'

    def test_pickle(self):
        if False:
            print('Hello World!')
        'Test pickling and unpickling of Hit.'
        buf = BytesIO()
        pickle.dump(self.hit, buf)
        unp = pickle.loads(buf.getvalue())
        self.compare_search_obj(self.hit, unp)

    def test_init_none(self):
        if False:
            i = 10
            return i + 15
        'Test Hit.__init__, no arguments.'
        hit = Hit()
        self.assertIsNone(hit.id)
        self.assertIsNone(hit.description)
        self.assertIsNone(hit.query_id)
        self.assertIsNone(hit.query_description)

    def test_init_id_only(self):
        if False:
            print('Hello World!')
        'Test Hit.__init__, with ID only.'
        hit = Hit(id='hit1')
        self.assertEqual('hit1', hit.id)
        self.assertIsNone(hit.description)
        self.assertIsNone(hit.query_id)
        self.assertIsNone(hit.query_description)

    def test_init_hsps_only(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Hit.__init__, with hsps only.'
        hit = Hit([hsp111, hsp112, hsp113])
        self.assertEqual('hit1', hit.id)
        self.assertEqual('<unknown description>', hit.description)
        self.assertEqual('query1', hit.query_id)
        self.assertEqual('<unknown description>', hit.query_description)

    def test_repr(self):
        if False:
            print('Hello World!')
        'Test Hit.__repr__.'
        self.assertEqual("Hit(id='hit1', query_id='query1', 3 hsps)", repr(self.hit))

    def test_hsps(self):
        if False:
            print('Hello World!')
        'Test Hit.hsps.'
        self.assertEqual([hsp111, hsp112, hsp113], self.hit.hsps)

    def test_fragments(self):
        if False:
            print('Hello World!')
        'Test Hit.fragments.'
        self.assertEqual([frag111, frag112, frag113, frag113b], self.hit.fragments)

    def test_iter(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Hit.__iter__.'
        for (counter, hsp) in enumerate(self.hit):
            self.assertIn(hsp, [hsp111, hsp112, hsp113])
        self.assertEqual(2, counter)

    def test_len(self):
        if False:
            i = 10
            return i + 15
        'Test Hit.__len__.'
        self.assertEqual(3, len(self.hit))

    def test_bool(self):
        if False:
            i = 10
            return i + 15
        'Test Hit.__bool__.'
        self.assertTrue(self.hit)

    def test_setitem_single(self):
        if False:
            return 10
        'Test Hit.__setitem__, single item.'
        self.hit[1] = hsp114
        self.assertEqual(self.hit.hsps, [hsp111, hsp114, hsp113])

    def test_item_multiple(self):
        if False:
            return 10
        'Test Hit.__setitem__, multiple items.'
        self.hit[:] = [hsp113, hsp112, hsp111]
        self.assertEqual(self.hit.hsps, [hsp113, hsp112, hsp111])

    def test_getitem_single(self):
        if False:
            return 10
        'Test Hit.__getitem__, single item.'
        hsp1 = self.hit[0]
        self.assertEqual(hsp111, hsp1)
        hsp3 = self.hit[-1]
        self.assertEqual(hsp113, hsp3)

    def test_getitem_multiple(self):
        if False:
            print('Hello World!')
        'Test Hit.__getitem__, multiple items.'
        new_hit = self.hit[:2]
        self.assertEqual(2, len(new_hit))
        self.assertEqual([hsp111, hsp112], new_hit.hsps)
        self.assertEqual(self.hit.id, new_hit.id)
        self.assertEqual(self.hit.query_id, new_hit.query_id)
        self.assertEqual(5e-10, new_hit.evalue)
        self.assertEqual('test', new_hit.name)

    def test_delitem(self):
        if False:
            return 10
        'Test Hit.__delitem__.'
        del self.hit[0]
        self.assertEqual(2, len(self.hit))
        self.assertEqual([hsp112, hsp113], self.hit.hsps)

    def test_validate_hsp_ok(self):
        if False:
            return 10
        'Test Hit._validate_hsp.'
        self.assertIsNone(self.hit._validate_hsp(hsp114))

    def test_validate_hsp_wrong_type(self):
        if False:
            return 10
        'Test Hit._validate_hsp, wrong type.'
        self.assertRaises(TypeError, self.hit._validate_hsp, 1)
        self.assertRaises(TypeError, self.hit._validate_hsp, Seq(''))

    def test_validate_hsp_wrong_query_id(self):
        if False:
            i = 10
            return i + 15
        'Test Hit._validate_hsp, wrong query ID.'
        self.assertRaises(ValueError, self.hit._validate_hsp, hsp211)

    def test_validate_hsp_wrong_hit_id(self):
        if False:
            i = 10
            return i + 15
        'Test Hit._validate_hsp, wrong hit ID.'
        self.assertRaises(ValueError, self.hit._validate_hsp, hsp121)

    def test_desc_set(self):
        if False:
            return 10
        'Test Hit.description setter.'
        hit = deepcopy(self.hit)
        new_desc = 'unicorn hox homolog'
        for hsp in hit:
            self.assertNotEqual(new_desc, hsp.hit_description)
            for fragment in hsp:
                self.assertNotEqual(new_desc, fragment.hit_description)
                self.assertNotEqual(new_desc, fragment.hit.description)
        hit.description = new_desc
        for hsp in hit:
            self.assertEqual(new_desc, hsp.hit_description)
            for fragment in hsp:
                self.assertEqual(new_desc, fragment.hit_description)
                self.assertEqual(new_desc, fragment.hit.description)

    def test_desc_set_no_seqrecord(self):
        if False:
            while True:
                i = 10
        'Test Hit.description setter, without HSP SeqRecords.'
        frag1 = HSPFragment('hit1', 'query')
        frag2 = HSPFragment('hit1', 'query')
        hit = Hit([HSP([x]) for x in [frag1, frag2]])
        new_desc = 'unicorn hox homolog'
        self.assertEqual(hit.description, '<unknown description>')
        for hsp in hit:
            self.assertEqual(hsp.hit_description, '<unknown description>')
            for fragment in hsp:
                self.assertEqual(hsp.hit_description, '<unknown description>')
        hit.description = new_desc
        self.assertEqual(hit.description, new_desc)
        for hsp in hit:
            self.assertTrue(hsp.hit_description, new_desc)
            for fragment in hsp:
                self.assertEqual(hsp.hit_description, new_desc)

    def test_id_set(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Hit.id setter.'
        hit = deepcopy(self.hit)
        self.assertEqual('hit1', hit.id)
        for hsp in hit.hsps:
            self.assertEqual('hit1', hsp.hit_id)
            for fragment in hsp:
                self.assertEqual(fragment.hit_id, 'hit1')
                self.assertEqual(fragment.hit.id, 'hit1')
        hit.id = 'new_id'
        self.assertEqual('new_id', hit.id)
        for hsp in hit.hsps:
            self.assertEqual('new_id', hsp.hit_id)
            for fragment in hsp:
                self.assertEqual(fragment.hit_id, 'new_id')
                self.assertEqual(fragment.hit.id, 'new_id')

    def test_append(self):
        if False:
            i = 10
            return i + 15
        'Test Hit.append.'
        self.hit.append(hsp114)
        self.assertEqual(4, len(self.hit))
        self.assertEqual(hsp114, self.hit[-1])

    def test_filter(self):
        if False:
            while True:
                i = 10
        'Test Hit.filter.'
        self.assertEqual([hsp111, hsp112, hsp113], self.hit.hsps)
        filter_func = lambda hsp: len(hsp[0]) >= 4
        filtered = self.hit.filter(filter_func)
        self.assertEqual([hsp111, hsp113], filtered.hsps)
        self.assertTrue(all((filter_func(hit) for hit in filtered)))
        self.assertEqual(5e-10, filtered.evalue)
        self.assertEqual('test', filtered.name)

    def test_filter_no_func(self):
        if False:
            i = 10
            return i + 15
        'Test Hit.filter, without arguments.'
        filtered = self.hit.filter()
        self.compare_search_obj(filtered, self.hit)
        self.assertNotEqual(id(filtered), id(self.hit))
        self.assertEqual(5e-10, filtered.evalue)
        self.assertEqual('test', filtered.name)

    def test_filter_no_filtered(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Hit.hit_filter, all hits filtered out.'
        filter_func = lambda hsp: len(hsp[0]) > 50
        filtered = self.hit.filter(filter_func)
        self.assertIsNone(filtered)

    def test_index(self):
        if False:
            return 10
        'Test Hit.index.'
        self.assertEqual(1, self.hit.index(hsp112))

    def test_index_not_present(self):
        if False:
            i = 10
            return i + 15
        'Test Hit.index, when index is not present.'
        self.assertRaises(ValueError, self.hit.index, hsp114)

    def test_map(self):
        if False:
            while True:
                i = 10
        'Test Hit.hsp_map.'
        hit = deepcopy(self.hit)
        for hsp in hit:
            setattr(hsp, 'mock', 13)

        def map_func(hsp):
            if False:
                print('Hello World!')
            mapped_frags = [x[1:] for x in hsp]
            return HSP(mapped_frags)
        mapped = hit.map(map_func)
        for hsp in mapped:
            self.assertFalse(hasattr(hsp, 'mock'))
        self.assertEqual('TGCGCAT', mapped[0][0].hit.seq)
        self.assertEqual('TGCGCAT', mapped[0][0].query.seq)
        self.assertEqual('TG', mapped[1][0].hit.seq)
        self.assertEqual('AT', mapped[1][0].query.seq)
        self.assertEqual('TTCG', mapped[2][0].hit.seq)
        self.assertEqual('T-CG', mapped[2][0].query.seq)
        self.assertEqual('TTCG', mapped[2][1].hit.seq)
        self.assertEqual('T-CG', mapped[2][1].query.seq)
        self.assertEqual(5e-10, mapped.evalue)
        self.assertEqual('test', mapped.name)

    def test_hsp_map_no_func(self):
        if False:
            print('Hello World!')
        'Test Hit.map, without arguments.'
        mapped = self.hit.map()
        self.compare_search_obj(mapped, self.hit)
        self.assertNotEqual(id(mapped), id(self.hit))
        self.assertEqual(5e-10, mapped.evalue)
        self.assertEqual('test', mapped.name)

    def test_pop(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Hit.pop.'
        self.assertEqual(hsp113, self.hit.pop())
        self.assertEqual(hsp111, self.hit.pop(0))

    def test_sort(self):
        if False:
            return 10
        'Test Hit.sort.'
        self.assertEqual([hsp111, hsp112, hsp113], self.hit.hsps)
        key = lambda batch_hsp: len(batch_hsp[0])
        self.hit.sort(key=key)
        self.assertEqual([hsp112, hsp113, hsp111], self.hit.hsps)

    def test_sort_not_in_place(self):
        if False:
            return 10
        'Test Hit.sort, not in place.'
        self.assertEqual([hsp111, hsp112, hsp113], self.hit.hsps)
        key = lambda hsp: len(hsp[0])
        sorted_hit = self.hit.sort(key=key, in_place=False)
        self.assertEqual([hsp112, hsp113, hsp111], sorted_hit.hsps)
        self.assertEqual([hsp111, hsp112, hsp113], self.hit.hsps)
        self.assertEqual(5e-10, sorted_hit.evalue)
        self.assertEqual('test', sorted_hit.name)

class HSPSingleFragmentCases(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.frag = HSPFragment('hit_id', 'query_id', 'ATCAGT', 'AT-ACT')
        self.frag.query_start = 0
        self.frag.query_end = 6
        self.frag.hit_start = 15
        self.frag.hit_end = 20
        self.hsp = HSP([self.frag])

    def test_init_no_fragment(self):
        if False:
            return 10
        'Test HSP.__init__ without fragments.'
        self.assertRaises(ValueError, HSP, [])

    def test_len(self):
        if False:
            return 10
        'Test HSP.__len__.'
        self.assertEqual(1, len(self.hsp))

    def test_fragment(self):
        if False:
            for i in range(10):
                print('nop')
        'Test HSP.fragment property.'
        self.assertIs(self.frag, self.hsp.fragment)

    def test_is_fragmented(self):
        if False:
            return 10
        'Test HSP.is_fragmented property.'
        self.assertFalse(self.hsp.is_fragmented)

    def test_seq(self):
        if False:
            while True:
                i = 10
        'Test HSP sequence properties.'
        self.assertEqual('ATCAGT', self.hsp.hit.seq)
        self.assertEqual('AT-ACT', self.hsp.query.seq)

    def test_alignment(self):
        if False:
            return 10
        'Test HSP.alignment property.'
        aln = self.hsp.aln
        self.assertIsInstance(aln, MultipleSeqAlignment)
        self.assertEqual(2, len(aln))
        self.assertTrue('ATCAGT', aln[0].seq)
        self.assertTrue('AT-ACT', aln[1].seq)

    def test_aln_span(self):
        if False:
            return 10
        'Test HSP.aln_span property.'
        self.assertEqual(6, self.hsp.aln_span)

    def test_span(self):
        if False:
            return 10
        'Test HSP span properties.'
        self.assertEqual(5, self.hsp.hit_span)
        self.assertEqual(6, self.hsp.query_span)

    def test_range(self):
        if False:
            return 10
        'Test HSP range properties.'
        self.assertEqual((15, 20), self.hsp.hit_range)
        self.assertEqual((0, 6), self.hsp.query_range)

    def test_setters_readonly(self):
        if False:
            return 10
        'Test HSP read-only properties.'
        read_onlies = ('range', 'span', 'strand', 'frame', 'start', 'end')
        for seq_type in ('query', 'hit'):
            self.assertRaises(AttributeError, setattr, self.hsp, seq_type, 'A')
            for attr in read_onlies:
                self.assertRaises(AttributeError, setattr, self.hsp, f'{seq_type}_{attr}', 5)
        self.assertRaises(AttributeError, setattr, self.hsp, 'aln', None)

class HSPMultipleFragmentCases(SearchTestBaseClass):

    def setUp(self):
        if False:
            print('Hello World!')
        self.frag1 = HSPFragment('hit_id', 'query_id', 'ATCAGT', 'AT-ACT')
        self.frag1.query_start = 0
        self.frag1.query_end = 6
        self.frag1.hit_start = 15
        self.frag1.hit_end = 20
        self.frag2 = HSPFragment('hit_id', 'query_id', 'GGG', 'CCC')
        self.frag2.query_start = 10
        self.frag2.query_end = 13
        self.frag2.hit_start = 158
        self.frag2.hit_end = 161
        self.hsp = HSP([self.frag1, self.frag2])

    def test_pickle(self):
        if False:
            print('Hello World!')
        'Test pickling and unpickling of HSP.'
        buf = BytesIO()
        pickle.dump(self.hsp, buf)
        unp = pickle.loads(buf.getvalue())
        self.compare_search_obj(self.hsp, unp)

    def test_len(self):
        if False:
            for i in range(10):
                print('nop')
        'Test HSP.__len__.'
        self.assertEqual(2, len(self.hsp))

    def test_getitem(self):
        if False:
            i = 10
            return i + 15
        'Test HSP.__getitem__.'
        self.assertIs(self.frag1, self.hsp[0])
        self.assertIs(self.frag2, self.hsp[1])

    def test_setitem_single(self):
        if False:
            while True:
                i = 10
        'Test HSP.__setitem__, single item.'
        frag3 = HSPFragment('hit_id', 'query_id', 'AAA', 'AAT')
        self.hsp[1] = frag3
        self.assertEqual(2, len(self.hsp))
        self.assertIs(self.frag1, self.hsp[0])
        self.assertIs(frag3, self.hsp[1])

    def test_setitem_multiple(self):
        if False:
            i = 10
            return i + 15
        'Test HSP.__setitem__, multiple items.'
        frag3 = HSPFragment('hit_id', 'query_id', 'AAA', 'AAT')
        frag4 = HSPFragment('hit_id', 'query_id', 'GGG', 'GAG')
        self.hsp[:2] = [frag3, frag4]
        self.assertEqual(2, len(self.hsp))
        self.assertIs(frag3, self.hsp[0])
        self.assertIs(frag4, self.hsp[1])

    def test_delitem(self):
        if False:
            return 10
        'Test HSP.__delitem__.'
        del self.hsp[0]
        self.assertEqual(1, len(self.hsp))
        self.assertIs(self.frag2, self.hsp[0])

    def test_contains(self):
        if False:
            return 10
        'Test HSP.__contains__.'
        frag3 = HSPFragment('hit_id', 'query_id', 'AAA', 'AAT')
        self.assertIn(self.frag1, self.hsp)
        self.assertNotIn(frag3, self.hsp)

    def test_fragments(self):
        if False:
            i = 10
            return i + 15
        'Test HSP.fragments property.'
        self.assertEqual([self.frag1, self.frag2], self.hsp.fragments)

    def test_is_fragmented(self):
        if False:
            print('Hello World!')
        'Test HSP.is_fragmented property.'
        self.assertTrue(self.hsp.is_fragmented)

    def test_seqs(self):
        if False:
            i = 10
            return i + 15
        'Test HSP sequence properties.'
        self.assertEqual(['ATCAGT', 'GGG'], [x.seq for x in self.hsp.hit_all])
        self.assertEqual(['AT-ACT', 'CCC'], [x.seq for x in self.hsp.query_all])

    def test_id_desc_set(self):
        if False:
            print('Hello World!')
        'Test HSP query and hit id and description setters.'
        for seq_type in ('query', 'hit'):
            for attr in ('id', 'description'):
                attr_name = f'{seq_type}_{attr}'
                value = getattr(self.hsp, attr_name)
                if attr == 'id':
                    self.assertEqual(value, attr_name)
                    for fragment in self.hsp:
                        self.assertEqual(getattr(fragment, attr_name), attr_name)
                else:
                    self.assertEqual(value, '<unknown description>')
                    for fragment in self.hsp:
                        self.assertEqual(getattr(fragment, attr_name), '<unknown description>')
                new_value = 'new_' + value
                setattr(self.hsp, attr_name, new_value)
                self.assertEqual(getattr(self.hsp, attr_name), new_value)
                self.assertNotEqual(getattr(self.hsp, attr_name), value)
                for fragment in self.hsp:
                    self.assertEqual(getattr(fragment, attr_name), new_value)
                    self.assertNotEqual(getattr(fragment, attr_name), value)

    def test_molecule_type(self):
        if False:
            while True:
                i = 10
        'Test HSP.molecule_type getter.'
        self.assertIsNone(self.hsp.molecule_type)

    def test_molecule_type_set(self):
        if False:
            print('Hello World!')
        'Test HSP.molecule_type setter.'
        self.assertIsNone(self.hsp.molecule_type)
        for frag in self.hsp.fragments:
            self.assertIsNone(frag.molecule_type)
        self.hsp.molecule_type = 'DNA'
        self.assertEqual(self.hsp.molecule_type, 'DNA')
        for frag in self.hsp.fragments:
            self.assertEqual(frag.molecule_type, 'DNA')

    def test_range(self):
        if False:
            i = 10
            return i + 15
        'Test HSP range properties.'
        self.assertEqual((15, 161), self.hsp.hit_range)
        self.assertEqual((0, 13), self.hsp.query_range)

    def test_ranges(self):
        if False:
            return 10
        'Test HSP ranges properties.'
        self.assertEqual([(15, 20), (158, 161)], self.hsp.hit_range_all)
        self.assertEqual([(0, 6), (10, 13)], self.hsp.query_range_all)

    def test_span(self):
        if False:
            for i in range(10):
                print('nop')
        'Test HSP span properties.'
        self.assertEqual(146, self.hsp.hit_span)
        self.assertEqual(13, self.hsp.query_span)

    def test_setters_readonly(self):
        if False:
            return 10
        'Test HSP read-only properties.'
        read_onlies = ('range_all', 'strand_all', 'frame_all')
        for seq_type in ('query', 'hit'):
            for attr in read_onlies:
                self.assertRaises(AttributeError, setattr, self.hsp, f'{seq_type}_{attr}', 5)
        self.assertRaises(AttributeError, setattr, self.hsp, 'aln_all', None)
        self.assertRaises(AttributeError, setattr, self.hsp, 'hit_all', None)
        self.assertRaises(AttributeError, setattr, self.hsp, 'query_all', None)

class HSPFragmentWithoutSeqCases(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fragment = HSPFragment('hit_id', 'query_id')

    def test_init(self):
        if False:
            print('Hello World!')
        'Test HSPFragment.__init__ attributes.'
        fragment = HSPFragment('hit_id', 'query_id')
        for seq_type in ('query', 'hit'):
            self.assertIsNone(getattr(fragment, seq_type))
            for attr in ('strand', 'frame', 'start', 'end'):
                attr_name = f'{seq_type}_{attr}'
                self.assertIsNone(getattr(fragment, attr_name))
        self.assertIsNone(fragment.aln)
        self.assertIsNone(fragment.molecule_type)
        self.assertEqual(fragment.aln_annotation, {})

    def test_seqmodel(self):
        if False:
            i = 10
            return i + 15
        'Test HSPFragment sequence attributes, no alignments.'
        self.assertIsNone(self.fragment.query)
        self.assertIsNone(self.fragment.hit)
        self.assertIsNone(self.fragment.aln)

    def test_len(self):
        if False:
            print('Hello World!')
        'Test HSPFragment.__len__, no alignments.'
        self.assertRaises(TypeError, len, self)
        self.fragment.aln_span = 5
        self.assertEqual(5, len(self.fragment))

    def test_repr(self):
        if False:
            while True:
                i = 10
        'Test HSPFragment.__repr__, no alignments.'
        self.assertEqual("HSPFragment(hit_id='hit_id', query_id='query_id')", repr(self.fragment))
        self.fragment.aln_span = 5
        self.assertEqual("HSPFragment(hit_id='hit_id', query_id='query_id', 5 columns)", repr(self.fragment))

    def test_getitem(self):
        if False:
            print('Hello World!')
        'Test HSPFragment.__getitem__, no alignments.'
        self.assertRaises(TypeError, self.fragment.__getitem__, 0)
        self.assertRaises(TypeError, self.fragment.__getitem__, slice(0, 2))

    def test_getitem_only_query(self):
        if False:
            i = 10
            return i + 15
        'Test HSPFragment.__getitem__, only query.'
        self.fragment.query = 'AATCG'
        self.assertEqual('ATCG', self.fragment[1:].query.seq)

    def test_getitem_only_hit(self):
        if False:
            i = 10
            return i + 15
        'Test HSPFragment.__getitem__, only hit.'
        self.fragment.hit = 'CATGC'
        self.assertEqual('ATGC', self.fragment[1:].hit.seq)

    def test_iter(self):
        if False:
            while True:
                i = 10
        'Test HSP.__iter__, no alignments.'
        self.assertRaises(TypeError, iter, self)

class HSPFragmentCases(SearchTestBaseClass):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fragment = HSPFragment('hit_id', 'query_id', 'ATGCTAGCTACA', 'ATG--AGCTAGG')

    def test_pickle(self):
        if False:
            return 10
        'Test pickling and unpickling of HSPFragment.'
        buf = BytesIO()
        pickle.dump(self.fragment, buf)
        unp = pickle.loads(buf.getvalue())
        self.compare_search_obj(self.fragment, unp)

    def test_init_with_seqrecord(self):
        if False:
            while True:
                i = 10
        'Test HSPFragment.__init__, with SeqRecord.'
        hit_seq = SeqRecord(Seq('ATGCTAGCTACA'))
        query_seq = SeqRecord(Seq('ATG--AGCTAGG'))
        hsp = HSPFragment('hit_id', 'query_id', hit_seq, query_seq)
        self.assertIsInstance(hsp.query, SeqRecord)
        self.assertIsInstance(hsp.hit, SeqRecord)
        self.assertIsInstance(hsp.aln, MultipleSeqAlignment)

    def test_init_wrong_seqtypes(self):
        if False:
            while True:
                i = 10
        'Test HSPFragment.__init__, wrong sequence argument types.'
        wrong_query = Seq('ATGC')
        wrong_hit = Seq('ATGC')
        self.assertRaises(TypeError, HSPFragment, 'hit_id', 'query_id', wrong_hit, wrong_query)

    def test_seqmodel(self):
        if False:
            for i in range(10):
                print('nop')
        'Test HSPFragment sequence attribute types and default values.'
        self.assertIsInstance(self.fragment.hit, SeqRecord)
        self.assertEqual('<unknown description>', self.fragment.hit.description)
        self.assertEqual('aligned hit sequence', self.fragment.hit.name)
        self.assertIsNone(self.fragment.hit.annotations['molecule_type'])
        self.assertIsInstance(self.fragment.query, SeqRecord)
        self.assertEqual('<unknown description>', self.fragment.query.description)
        self.assertEqual('aligned query sequence', self.fragment.query.name)
        self.assertIsNone(self.fragment.query.annotations['molecule_type'])
        self.assertIsInstance(self.fragment.aln, MultipleSeqAlignment)
        with self.assertRaises(AttributeError):
            self.fragment.aln.molecule_type

    def test_molecule_type_no_seq(self):
        if False:
            while True:
                i = 10
        'Test HSPFragment molecule_type property, query and hit sequences not present.'
        self.assertIsNone(self.fragment.molecule_type)
        self.fragment.molecule_type = 'DNA'
        self.assertEqual(self.fragment.molecule_type, 'DNA')

    def test_molecule_type_with_seq(self):
        if False:
            for i in range(10):
                print('nop')
        'Test HSPFragment molecule_type property, query or hit sequences present.'
        self.assertIsNone(self.fragment.molecule_type)
        self.fragment._hit = SeqRecord(Seq('AAA'))
        self.fragment._query = SeqRecord(Seq('AAA'))
        self.fragment.molecule_type = 'DNA'
        self.assertEqual(self.fragment.molecule_type, 'DNA')
        self.assertEqual(self.fragment.hit.annotations['molecule_type'], 'DNA')
        self.assertEqual(self.fragment.query.annotations['molecule_type'], 'DNA')

    def test_seq_unequal_hit_query_len(self):
        if False:
            while True:
                i = 10
        'Test HSPFragment sequence setter with unequal hit and query lengths.'
        for seq_type in ('hit', 'query'):
            opp_type = 'query' if seq_type == 'hit' else 'hit'
            fragment = HSPFragment('hit_id', 'query_id')
            setattr(fragment, seq_type, 'ATGCACAACAGGA')
            self.assertRaises(ValueError, setattr, fragment, opp_type, 'ATGCGA')

    def test_len(self):
        if False:
            return 10
        'Test HSPFragment.__len__.'
        self.assertEqual(12, len(self.fragment))

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        'Test HSPFragment.__repr__.'
        self.assertEqual("HSPFragment(hit_id='hit_id', query_id='query_id', 12 columns)", repr(self.fragment))

    def test_getitem(self):
        if False:
            while True:
                i = 10
        'Test HSPFragment.__getitem__.'
        sliced_fragment = self.fragment[:5]
        self.assertIsInstance(sliced_fragment, HSPFragment)
        self.assertEqual(5, len(sliced_fragment))
        self.assertEqual('ATGCT', sliced_fragment.hit.seq)
        self.assertEqual('ATG--', sliced_fragment.query.seq)

    def test_getitem_attrs(self):
        if False:
            return 10
        'Test HSPFragment.__getitem__, with attributes.'
        setattr(self.fragment, 'attr_original', 1000)
        setattr(self.fragment, 'hit_description', 'yeah')
        setattr(self.fragment, 'hit_strand', 1)
        setattr(self.fragment, 'query_frame', None)
        self.assertEqual(1000, getattr(self.fragment, 'attr_original'))
        self.assertEqual('yeah', getattr(self.fragment, 'hit_description'))
        self.assertEqual(1, getattr(self.fragment, 'hit_strand'))
        self.assertIsNone(getattr(self.fragment, 'query_frame'))
        new_hsp = self.fragment[:5]
        self.assertFalse(hasattr(new_hsp, 'attr_original'))
        self.assertEqual(1000, getattr(self.fragment, 'attr_original'))
        self.assertEqual('yeah', getattr(self.fragment, 'hit_description'))
        self.assertEqual(1, getattr(self.fragment, 'hit_strand'))
        self.assertIsNone(getattr(self.fragment, 'query_frame'))

    def test_getitem_alignment_annot(self):
        if False:
            for i in range(10):
                print('nop')
        'Test HSPFragment.__getitem__, with alignment annotation.'
        setattr(self.fragment, 'aln_annotation', {'test': '182718738172'})
        new_hsp = self.fragment[:5]
        self.assertEqual('18271', new_hsp.aln_annotation['test'])

    def test_default_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        "Test HSPFragment attributes' default values."
        fragment = HSPFragment()
        self.assertEqual('<unknown id>', fragment.hit_id)
        self.assertEqual('<unknown id>', fragment.query_id)
        self.assertEqual('<unknown description>', fragment.hit_description)
        self.assertEqual('<unknown description>', fragment.query_description)
        self.assertIsNone(fragment.hit)
        self.assertIsNone(fragment.query)
        self.assertIsNone(fragment.aln)
        self.assertEqual([], fragment.hit_features)
        self.assertEqual([], fragment.query_features)
        self.assertIsNone(fragment.hit_strand)
        self.assertIsNone(fragment.query_strand)
        self.assertIsNone(fragment.hit_frame)
        self.assertIsNone(fragment.query_frame)

    def test_id_desc_set(self):
        if False:
            return 10
        'Test HSPFragment query and hit id and description setters.'
        for seq_type in ('query', 'hit'):
            for attr in ('id', 'description'):
                attr_name = f'{seq_type}_{attr}'
                value = getattr(self.fragment, attr_name)
                if attr == 'id':
                    self.assertEqual(value, attr_name)
                else:
                    self.assertEqual(value, '<unknown description>')
                new_value = 'new_' + value
                setattr(self.fragment, attr_name, new_value)
                self.assertEqual(getattr(self.fragment, attr_name), new_value)
                self.assertNotEqual(getattr(self.fragment, attr_name), value)

    def test_frame_set_ok(self):
        if False:
            for i in range(10):
                print('nop')
        'Test HSPFragment query and hit frame setters.'
        attr = 'frame'
        for seq_type in ('query', 'hit'):
            attr_name = f'{seq_type}_{attr}'
            for value in (-3, -2, -1, 0, 1, 2, 3, None):
                setattr(self.fragment, attr_name, value)
                self.assertEqual(value, getattr(self.fragment, attr_name))

    def test_frame_set_error(self):
        if False:
            print('Hello World!')
        'Test HSPFragment query and hit frame setters, invalid values.'
        attr = 'frame'
        for seq_type in ('query', 'hit'):
            func_name = f'_{seq_type}_{attr}_set'
            func = getattr(self.fragment, func_name)
            for value in ('3', '+3', '-2', 'plus'):
                self.assertRaises(ValueError, func, value)

    def test_strand_set_ok(self):
        if False:
            i = 10
            return i + 15
        'Test HSPFragment query and hit strand setters.'
        attr = 'strand'
        for seq_type in ('query', 'hit'):
            attr_name = f'{seq_type}_{attr}'
            for value in (-1, 0, 1, None):
                setattr(self.fragment, attr_name, value)
                self.assertEqual(value, getattr(self.fragment, attr_name))

    def test_strand_set_error(self):
        if False:
            i = 10
            return i + 15
        'Test HSPFragment query and hit strand setters, invalid values.'
        attr = 'strand'
        for seq_type in ('query', 'hit'):
            func_name = f'_{seq_type}_{attr}_set'
            func = getattr(self.fragment, func_name)
            for value in (3, 'plus', 'minus', '-', '+'):
                self.assertRaises(ValueError, func, value)

    def test_strand_set_from_plus_frame(self):
        if False:
            print('Hello World!')
        'Test HSPFragment query and hit strand getters, from plus frame.'
        for seq_type in ('query', 'hit'):
            attr_name = f'{seq_type}_strand'
            self.assertIsNone(getattr(self.fragment, attr_name))
            setattr(self.fragment, f'{seq_type}_frame', 3)
            self.assertEqual(1, getattr(self.fragment, attr_name))

    def test_strand_set_from_minus_frame(self):
        if False:
            i = 10
            return i + 15
        'Test HSPFragment query and hit strand getters, from minus frame.'
        for seq_type in ('query', 'hit'):
            attr_name = f'{seq_type}_strand'
            self.assertIsNone(getattr(self.fragment, attr_name))
            setattr(self.fragment, f'{seq_type}_frame', -2)
            self.assertEqual(-1, getattr(self.fragment, attr_name))

    def test_strand_set_from_zero_frame(self):
        if False:
            print('Hello World!')
        'Test HSPFragment query and hit strand getters, from zero frame.'
        for seq_type in ('query', 'hit'):
            attr_name = f'{seq_type}_strand'
            self.assertIsNone(getattr(self.fragment, attr_name))
            setattr(self.fragment, f'{seq_type}_frame', 0)
            self.assertEqual(0, getattr(self.fragment, attr_name))

    def test_coords_setters_getters(self):
        if False:
            while True:
                i = 10
        'Test HSPFragment query and hit coordinate-related setters and getters.'
        for seq_type in ('query', 'hit'):
            attr_start = f"{seq_type}_{'start'}"
            attr_end = f"{seq_type}_{'end'}"
            setattr(self.fragment, attr_start, 9)
            setattr(self.fragment, attr_end, 99)
            span = getattr(self.fragment, f'{seq_type}_span')
            self.assertEqual(90, span)
            range = getattr(self.fragment, f'{seq_type}_range')
            self.assertEqual((9, 99), range)

    def test_coords_setters_readonly(self):
        if False:
            i = 10
            return i + 15
        'Test HSPFragment query and hit coordinate-related read-only getters.'
        read_onlies = ('range', 'span')
        for seq_type in ('query', 'hit'):
            for attr in read_onlies:
                self.assertRaises(AttributeError, setattr, self.fragment, f'{seq_type}_{attr}', 5)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)