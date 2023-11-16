from __future__ import absolute_import, division, print_function, unicode_literals
'\n    sockjs.tornado.transports.xhrstreaming\n    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n    Xhr-Streaming transport implementation\n'
from octoprint.vendor.sockjs.tornado.transports import streamingbase
from octoprint.vendor.sockjs.tornado.util import no_auto_finish

class XhrStreamingTransport(streamingbase.StreamingTransportBase):
    name = 'xhr_streaming'

    @no_auto_finish
    def post(self, session_id):
        if False:
            for i in range(10):
                print('nop')
        self.preflight()
        self.handle_session_cookie()
        self.disable_cache()
        self.set_header('Content-Type', 'application/javascript; charset=UTF-8')
        self.write('h' * 2048 + '\n')
        self.flush()
        if not self._attach_session(session_id, False):
            self.finish()
            return
        if self.session:
            self.session.flush()

    def send_pack(self, message, binary=False):
        if False:
            i = 10
            return i + 15
        if binary:
            raise Exception('binary not supported for XhrStreamingTransport')
        self.active = False
        try:
            self.notify_sent(len(message))
            self.write(message + '\n')
            self.flush().add_done_callback(self.send_complete)
        except IOError:
            self.session.delayed_close()
            self._detach()