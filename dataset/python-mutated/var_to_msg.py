from gnuradio import gr
import pmt

class var_to_msg_pair(gr.sync_block):
    """
    This block has a callback that will emit a message pair with the updated variable
    value when called. This is useful for monitoring a GRC variable and emitting a message
    when it is changed.
    """

    def __init__(self, pairname):
        if False:
            i = 10
            return i + 15
        gr.sync_block.__init__(self, name='var_to_msg_pair', in_sig=None, out_sig=None)
        self.pairname = pairname
        self.message_port_register_out(pmt.intern('msgout'))

    def variable_changed(self, value):
        if False:
            print('Hello World!')
        try:
            if type(value) == float:
                p = pmt.from_double(value)
            elif type(value) == int:
                p = pmt.from_long(value)
            elif type(value) == bool:
                p = pmt.from_bool(value)
            elif type(value) == str:
                p = pmt.intern(value)
            else:
                p = pmt.to_pmt(value)
            self.message_port_pub(pmt.intern('msgout'), pmt.cons(pmt.intern(self.pairname), p))
        except Exception as e:
            gr.log.error('Unable to convert ' + repr(value) + ' to PDU, no message will be emitted (reason: %s)' % repr(e))

    def stop(self):
        if False:
            print('Hello World!')
        return True