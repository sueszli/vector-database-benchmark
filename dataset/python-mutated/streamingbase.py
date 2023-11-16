from __future__ import absolute_import, division, print_function, unicode_literals
from octoprint.vendor.sockjs.tornado.transports import pollingbase

class StreamingTransportBase(pollingbase.PollingTransportBase):

    def initialize(self, server):
        if False:
            return 10
        super(StreamingTransportBase, self).initialize(server)
        self.amount_limit = self.server.settings['response_limit']
        if hasattr(self.request, 'connection') and (not self.request.version == 'HTTP/1.1'):
            self.request.connection.no_keep_alive = True

    def notify_sent(self, data_len):
        if False:
            print('Hello World!')
        '\n            Update amount of data sent\n        '
        self.amount_limit -= data_len

    def should_finish(self):
        if False:
            print('Hello World!')
        '\n            Check if transport should close long running connection after\n            sending X bytes to the client.\n\n            `data_len`\n                Amount of data that was sent\n        '
        if self.amount_limit <= 0:
            return True
        return False

    def send_complete(self, f=None):
        if False:
            while True:
                i = 10
        '\n            Verify if connection should be closed based on amount of data that was sent.\n        '
        self.active = True
        if self.should_finish():
            self._detach()
            if not self._finished:
                self.safe_finish()
        elif self.session:
            self.session.flush()