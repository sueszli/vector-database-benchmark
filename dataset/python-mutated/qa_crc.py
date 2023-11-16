from gnuradio import gr, gr_unittest, blocks, digital
import pmt

class qa_crc(gr_unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Common part of all CRC tests\n\n        Creates a flowgraph, a Message Debug block, and a PDU\n        containing the numbers 0x00 through 0x0F.\n        '
        self.tb = gr.top_block()
        self.dbg = blocks.message_debug()
        self.data = list(range(16))
        self.pdu = pmt.cons(pmt.PMT_NIL, pmt.init_u8vector(len(self.data), self.data))

    def run_crc_append(self, crc_params, crc_result):
        if False:
            print('Hello World!')
        'Common part of CRC Append tests\n\n        Creates a CRC Append block with the specified crc_params parameters,\n        connects it to the Message Debug block, sends a test PDU to the\n        CRC Append block, and checks that the output PDU matches the expected\n        crc_result.\n        '
        crc_append_block = digital.crc_append(*crc_params)
        self.tb.msg_connect((crc_append_block, 'out'), (self.dbg, 'store'))
        crc_append_block.to_basic_block()._post(pmt.intern('in'), self.pdu)
        crc_append_block.to_basic_block()._post(pmt.intern('system'), pmt.cons(pmt.intern('done'), pmt.from_long(1)))
        self.tb.start()
        self.tb.wait()
        self.assertEqual(self.dbg.num_messages(), 1)
        out = pmt.u8vector_elements(pmt.cdr(self.dbg.get_message(0)))
        self.assertEqual(out[:len(self.data)], self.data)
        self.assertEqual(out[len(self.data):], crc_result)

    def common_test_crc_check(self, matching_crc, header_bytes=0):
        if False:
            i = 10
            return i + 15
        'Common part of CRC Check tests\n\n        Creates a CRC Append block and a CRC Check block using either the\n        same CRC or a different one depending on the whether matching_crc\n        is True or False. Connects CRC Append -> CRC Check -> Message Debug\n        and sends a PDU through. There are two message debugs to allow\n        checking whether the PDU ended up in the ok or fail port of the\n        CRC Check block.\n        '
        crc_append_block = digital.crc_append(16, 4129, 0, 0, False, False, False, header_bytes)
        x = 0 if matching_crc else 65535
        crc_check_block = digital.crc_check(16, 4129, x, x, False, False, False, True, header_bytes)
        self.dbg_fail = blocks.message_debug()
        self.tb.msg_connect((crc_append_block, 'out'), (crc_check_block, 'in'))
        self.tb.msg_connect((crc_check_block, 'ok'), (self.dbg, 'store'))
        self.tb.msg_connect((crc_check_block, 'fail'), (self.dbg_fail, 'store'))
        crc_append_block.to_basic_block()._post(pmt.intern('in'), self.pdu)
        crc_append_block.to_basic_block()._post(pmt.intern('system'), pmt.cons(pmt.intern('done'), pmt.from_long(1)))
        self.tb.start()
        self.tb.wait()

    def test_crc_check(self):
        if False:
            return 10
        'Test a successful CRC check\n\n        Checks that the PDU ends in the ok port of CRC check\n        '
        self.common_test_crc_check(matching_crc=True)
        self.assertEqual(self.dbg.num_messages(), 1)
        out = pmt.u8vector_elements(pmt.cdr(self.dbg.get_message(0)))
        self.assertEqual(out, self.data)
        self.assertEqual(self.dbg_fail.num_messages(), 0)

    def test_crc_check_header_bytes(self):
        if False:
            print('Hello World!')
        'Test a successful CRC check (skipping some header bytes)\n\n        Checks that the PDU ends in the ok port of CRC check\n        '
        self.common_test_crc_check(matching_crc=True, header_bytes=5)
        self.assertEqual(self.dbg.num_messages(), 1)
        out = pmt.u8vector_elements(pmt.cdr(self.dbg.get_message(0)))
        self.assertEqual(out, self.data)
        self.assertEqual(self.dbg_fail.num_messages(), 0)

    def test_crc_check_wrong_crc(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a failed CRC check\n\n        Checks that the PDU ends in the fail port of CRC check\n        '
        self.common_test_crc_check(matching_crc=False)
        self.assertEqual(self.dbg.num_messages(), 0)
        self.assertEqual(self.dbg_fail.num_messages(), 1)
        out = pmt.u8vector_elements(pmt.cdr(self.dbg_fail.get_message(0)))
        self.assertEqual(out, self.data)

    def test_crc_append_crc16_ccitt_zero(self):
        if False:
            for i in range(10):
                print('nop')
        'Test CRC-16-CCITT-Zero calculation'
        self.run_crc_append((16, 4129, 0, 0, False, False, False), [81, 61])

    def test_crc_append_crc16_ccitt_false(self):
        if False:
            i = 10
            return i + 15
        'Test CRC-16-CCITT-False calculation'
        self.run_crc_append((16, 4129, 65535, 0, False, False, False), [59, 55])

    def test_crc_append_crc16_ccitt_x25(self):
        if False:
            while True:
                i = 10
        'Test CRC-16-CCITT-X.25 calculation'
        self.run_crc_append((16, 4129, 65535, 65535, True, True, False), [19, 233])

    def test_crc_append_crc32(self):
        if False:
            for i in range(10):
                print('nop')
        'Test CRC-32 calculation'
        self.run_crc_append((32, 79764919, 4294967295, 4294967295, True, True, False), [206, 206, 226, 136])

    def test_crc_append_crc32c(self):
        if False:
            for i in range(10):
                print('nop')
        'Test CRC-32C calculation'
        self.run_crc_append((32, 517762881, 4294967295, 4294967295, True, True, False), [217, 201, 8, 235])

    def test_crc_append_crc32c_endianness_swap(self):
        if False:
            i = 10
            return i + 15
        'Test CRC-32C calculation with endianness swapped'
        self.run_crc_append((32, 517762881, 4294967295, 4294967295, True, True, True), [235, 8, 201, 217])

    def test_crc_append_crc32c_skip_header_bytes(self):
        if False:
            return 10
        'Test CRC-32C calculation skipping some header bytes'
        skip_bytes = 3
        self.run_crc_append((32, 517762881, 4294967295, 4294967295, True, True, False, skip_bytes), [232, 98, 96, 104])

class qa_crc_class(gr_unittest.TestCase):

    def test_crc_crc32c(self):
        if False:
            return 10
        'Test CRC-32C calculation (using crc class directly)'
        c = digital.crc(32, 517762881, 4294967295, 4294967295, True, True)
        out = c.compute(list(range(16)))
        self.assertEqual(c.compute(list(range(16))), 3653830891)
if __name__ == '__main__':
    gr_unittest.run(qa_crc)
    gr_unittest.run(qa_crc_class)