from gnuradio import gr, gr_unittest, blocks, pdu
import numpy as np
import pmt
import time

class qa_tags_to_pdu(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def tearDown(self):
        if False:
            while True:
                i = 10
        pass

    def test_001_simple(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()
        start_time = 0.1
        sob_tag = gr.tag_utils.python_to_tag((34, pmt.intern('SOB'), pmt.PMT_T, pmt.intern('src')))
        eob_tag = gr.tag_utils.python_to_tag((34 + 8 * 31, pmt.intern('EOB'), pmt.PMT_T, pmt.intern('src')))
        vs = blocks.vector_source_s(range(350), False, 1, [sob_tag, eob_tag])
        t2p = pdu.tags_to_pdu_s(pmt.intern('SOB'), pmt.intern('EOB'), 1024, 512000, [], False, 0, start_time)
        t2p.set_eob_parameters(8, 0)
        dbg = blocks.message_debug()
        self.tb.connect(vs, t2p)
        self.tb.msg_connect((t2p, 'pdus'), (dbg, 'store'))
        expected_vec = pmt.init_s16vector(8 * 31, range(34, 34 + 8 * 31))
        expected_time = start_time + 34 / 512000.0
        self.tb.run()
        self.assertEqual(dbg.num_messages(), 1)
        self.assertTrue(pmt.equal(pmt.cdr(dbg.get_message(0)), expected_vec))
        time_tuple1 = pmt.dict_ref(pmt.car(dbg.get_message(0)), pmt.intern('rx_time'), pmt.PMT_NIL)
        self.assertAlmostEqual(pmt.to_uint64(pmt.tuple_ref(time_tuple1, 0)) + pmt.to_double(pmt.tuple_ref(time_tuple1, 1)), expected_time)
        self.tb = None

    def test_002_secondSOB(self):
        if False:
            return 10
        self.tb = gr.top_block()
        start_time = 4.999999999
        sob_tag = gr.tag_utils.python_to_tag((34, pmt.intern('SOB'), pmt.PMT_T, pmt.intern('src')))
        sob_tag2 = gr.tag_utils.python_to_tag((51, pmt.intern('SOB'), pmt.PMT_T, pmt.intern('src')))
        eob_tag = gr.tag_utils.python_to_tag((51 + 8 * 26, pmt.intern('EOB'), pmt.PMT_T, pmt.intern('src')))
        vs = blocks.vector_source_s(range(350), False, 1, [sob_tag, sob_tag2, eob_tag])
        t2p = pdu.tags_to_pdu_s(pmt.intern('SOB'), pmt.intern('EOB'), 1024, 460800, [], False, 0, start_time)
        t2p.set_eob_parameters(8, 0)
        dbg = blocks.message_debug()
        self.tb.connect(vs, t2p)
        self.tb.msg_connect((t2p, 'pdus'), (dbg, 'store'))
        expected_vec = pmt.init_s16vector(8 * 26, range(51, 51 + 8 * 26))
        expected_time = start_time + 51 / 460800.0
        self.tb.run()
        self.assertEqual(dbg.num_messages(), 1)
        self.assertTrue(pmt.equal(pmt.cdr(dbg.get_message(0)), expected_vec))
        time_tuple1 = pmt.dict_ref(pmt.car(dbg.get_message(0)), pmt.intern('rx_time'), pmt.PMT_NIL)
        self.assertAlmostEqual(pmt.to_uint64(pmt.tuple_ref(time_tuple1, 0)) + pmt.to_double(pmt.tuple_ref(time_tuple1, 1)), expected_time)
        self.tb = None

    def test_003_double_eob_rej_tt_update(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()
        start_time = 0.0
        sob_tag = gr.tag_utils.python_to_tag((51, pmt.intern('SOB'), pmt.PMT_T, pmt.intern('src')))
        eob_tag = gr.tag_utils.python_to_tag((51 + 8 * 11, pmt.intern('EOB'), pmt.PMT_T, pmt.intern('src')))
        time_tuple = pmt.make_tuple(pmt.from_uint64(4), pmt.from_double(0.125), pmt.from_uint64(10000000), pmt.from_double(4000000.0))
        time_tag = gr.tag_utils.python_to_tag((360, pmt.intern('rx_time'), time_tuple, pmt.intern('src')))
        sob_tag2 = gr.tag_utils.python_to_tag((400, pmt.intern('SOB'), pmt.PMT_T, pmt.intern('src')))
        eob_tag2e = gr.tag_utils.python_to_tag((409, pmt.intern('EOB'), pmt.PMT_T, pmt.intern('src')))
        eob_tag2 = gr.tag_utils.python_to_tag((416, pmt.intern('EOB'), pmt.PMT_T, pmt.intern('src')))
        vs = blocks.vector_source_s(range(500), False, 1, [sob_tag, eob_tag, time_tag, sob_tag2, eob_tag2e, eob_tag2])
        t2p = pdu.tags_to_pdu_s(pmt.intern('SOB'), pmt.intern('EOB'), 1024, 1000000, [], False, 0, start_time)
        t2p.set_eob_parameters(8, 0)
        dbg = blocks.message_debug()
        self.tb.connect(vs, t2p)
        self.tb.msg_connect((t2p, 'pdus'), (dbg, 'store'))
        expected_vec1 = pmt.init_s16vector(8 * 11, range(51, 51 + 8 * 11))
        expected_vec2 = pmt.init_s16vector(16, list(range(400, 409)) + [0] * 7)
        expected_time1 = start_time + 51 / 1000000.0
        expected_time2 = 4.125 + (400 - 360) / 1000000.0
        self.tb.run()
        self.assertEqual(dbg.num_messages(), 2)
        self.assertTrue(pmt.equal(pmt.cdr(dbg.get_message(0)), expected_vec1))
        self.assertTrue(pmt.equal(pmt.cdr(dbg.get_message(1)), expected_vec2))
        time_tuple1 = pmt.dict_ref(pmt.car(dbg.get_message(0)), pmt.intern('rx_time'), pmt.PMT_NIL)
        time_tuple2 = pmt.dict_ref(pmt.car(dbg.get_message(1)), pmt.intern('rx_time'), pmt.PMT_NIL)
        self.assertAlmostEqual(pmt.to_uint64(pmt.tuple_ref(time_tuple1, 0)) + pmt.to_double(pmt.tuple_ref(time_tuple1, 1)), expected_time1)
        self.assertAlmostEqual(pmt.to_uint64(pmt.tuple_ref(time_tuple2, 0)) + pmt.to_double(pmt.tuple_ref(time_tuple2, 1)), expected_time2)
        self.tb = None

    def test_004_boost_time(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb = gr.top_block()
        start_time = 0.1
        sob_tag = gr.tag_utils.python_to_tag((34, pmt.intern('SOB'), pmt.PMT_T, pmt.intern('src')))
        eob_tag = gr.tag_utils.python_to_tag((34 + 8 * 31, pmt.intern('EOB'), pmt.PMT_T, pmt.intern('src')))
        vs = blocks.vector_source_s(range(350), False, 1, [sob_tag, eob_tag])
        t2p = pdu.tags_to_pdu_s(pmt.intern('SOB'), pmt.intern('EOB'), 1024, 512000, [], False, 0, start_time)
        t2p.enable_time_debug(True)
        t2p.set_eob_parameters(8, 0)
        dbg = blocks.message_debug()
        self.tb.connect(vs, t2p)
        self.tb.msg_connect((t2p, 'pdus'), (dbg, 'store'))
        expected_vec = pmt.init_s16vector(8 * 31, range(34, 34 + 8 * 31))
        expected_time = start_time + 34 / 512000.0
        ts = time.time()
        self.tb.run()
        self.assertEqual(dbg.num_messages(), 1)
        self.assertTrue(pmt.equal(pmt.cdr(dbg.get_message(0)), expected_vec))
        time_tuple1 = pmt.dict_ref(pmt.car(dbg.get_message(0)), pmt.intern('rx_time'), pmt.PMT_NIL)
        self.assertAlmostEqual(pmt.to_uint64(pmt.tuple_ref(time_tuple1, 0)) + pmt.to_double(pmt.tuple_ref(time_tuple1, 1)), expected_time)
        self.tb = None

    def test_005_two_sobs_misaligned(self):
        if False:
            i = 10
            return i + 15
        self.tb = gr.top_block()
        start_time = 0.1
        sob_tag = gr.tag_utils.python_to_tag((34, pmt.intern('SOB'), pmt.PMT_T, pmt.intern('src')))
        sob_tag2 = gr.tag_utils.python_to_tag((35, pmt.intern('SOB'), pmt.PMT_T, pmt.intern('src')))
        eob_tag = gr.tag_utils.python_to_tag((34 + 8 * 31, pmt.intern('EOB'), pmt.PMT_T, pmt.intern('src')))
        vs = blocks.vector_source_s(range(1350), False, 1, [sob_tag, sob_tag2, eob_tag])
        t2p = pdu.tags_to_pdu_s(pmt.intern('SOB'), pmt.intern('EOB'), 1024, 512000, [], False, 0, start_time)
        t2p.set_eob_parameters(8, 0)
        dbg = blocks.message_debug()
        self.tb.connect(vs, t2p)
        self.tb.msg_connect((t2p, 'pdus'), (dbg, 'store'))
        expected_vec = pmt.init_s16vector(8 * 31, list(range(35, 34 + 8 * 31)) + [0])
        expected_time = start_time + 35 / 512000.0
        self.tb.run()
        self.assertEqual(dbg.num_messages(), 1)
        self.assertTrue(pmt.equal(pmt.cdr(dbg.get_message(0)), expected_vec))
        time_tuple1 = pmt.dict_ref(pmt.car(dbg.get_message(0)), pmt.intern('rx_time'), pmt.PMT_NIL)
        self.assertAlmostEqual(pmt.to_uint64(pmt.tuple_ref(time_tuple1, 0)) + pmt.to_double(pmt.tuple_ref(time_tuple1, 1)), expected_time)
        self.tb = None

    def test_006_max_pdu_size(self):
        if False:
            while True:
                i = 10
        self.tb = gr.top_block()
        start_time = 0.1
        max_size = 100
        sob_tag = gr.tag_utils.python_to_tag((10, pmt.intern('SOB'), pmt.PMT_T, pmt.intern('src')))
        eob_tag = gr.tag_utils.python_to_tag((91, pmt.intern('EOB'), pmt.PMT_T, pmt.intern('src')))
        sob_tag3 = gr.tag_utils.python_to_tag((11 + max_size, pmt.intern('SOB'), pmt.PMT_T, pmt.intern('src')))
        vs = blocks.vector_source_s(range(1350), False, 1, [sob_tag, eob_tag, sob_tag3])
        t2p = pdu.tags_to_pdu_s(pmt.intern('SOB'), pmt.intern('EOB'), 1024, 512000, [], False, 0, start_time)
        t2p.set_eob_parameters(10, 0)
        t2p.set_max_pdu_size(max_size)
        dbg = blocks.message_debug()
        self.tb.connect(vs, t2p)
        self.tb.msg_connect((t2p, 'pdus'), (dbg, 'store'))
        expected_vec = pmt.init_s16vector(9 * 10, list(range(10, 91)) + [0] * 9)
        expected_time = start_time + 10 / 512000.0
        self.tb.run()
        self.assertEqual(dbg.num_messages(), 2)
        self.assertTrue(pmt.equal(pmt.cdr(dbg.get_message(0)), expected_vec))
        time_tuple1 = pmt.dict_ref(pmt.car(dbg.get_message(0)), pmt.intern('rx_time'), pmt.PMT_NIL)
        self.assertAlmostEqual(pmt.to_uint64(pmt.tuple_ref(time_tuple1, 0)) + pmt.to_double(pmt.tuple_ref(time_tuple1, 1)), expected_time)
        self.tb = None

    def test_007_max_pdu_size_SOBs(self):
        if False:
            return 10
        self.tb = gr.top_block()
        start_time = 0.1
        max_size = 100
        sob_tag = gr.tag_utils.python_to_tag((10, pmt.intern('SOB'), pmt.PMT_T, pmt.intern('src')))
        sob_tag3 = gr.tag_utils.python_to_tag((10 + max_size, pmt.intern('SOB'), pmt.PMT_T, pmt.intern('src')))
        vs = blocks.vector_source_s(range(1350), False, 1, [sob_tag, sob_tag3])
        t2p = pdu.tags_to_pdu_s(pmt.intern('SOB'), pmt.intern('EOB'), 1024, 512000, [], False, 0, start_time)
        t2p.set_eob_parameters(10, 0)
        t2p.set_max_pdu_size(max_size)
        dbg = blocks.message_debug()
        self.tb.connect(vs, t2p)
        self.tb.msg_connect((t2p, 'pdus'), (dbg, 'store'))
        expected_vec = pmt.init_s16vector(max_size, range(10, 10 + max_size))
        expected_time = start_time + 10 / 512000.0
        self.tb.run()
        self.assertEqual(dbg.num_messages(), 2)
        self.assertTrue(pmt.equal(pmt.cdr(dbg.get_message(0)), expected_vec))
        time_tuple1 = pmt.dict_ref(pmt.car(dbg.get_message(0)), pmt.intern('rx_time'), pmt.PMT_NIL)
        self.assertAlmostEqual(pmt.to_uint64(pmt.tuple_ref(time_tuple1, 0)) + pmt.to_double(pmt.tuple_ref(time_tuple1, 1)), expected_time)
        self.tb = None
if __name__ == '__main__':
    gr_unittest.run(qa_tags_to_pdu)