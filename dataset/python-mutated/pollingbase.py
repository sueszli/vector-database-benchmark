from __future__ import absolute_import, division, print_function, unicode_literals
'\n    sockjs.tornado.transports.pollingbase\n    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n    Polling transports base\n'
from octoprint.vendor.sockjs.tornado import basehandler
from octoprint.vendor.sockjs.tornado.transports import base

class PollingTransportBase(basehandler.PreflightHandler, base.BaseTransportMixin):
    """Polling transport handler base class"""

    def initialize(self, server):
        if False:
            return 10
        super(PollingTransportBase, self).initialize(server)
        self.session = None
        self.active = True

    def _get_session(self, session_id):
        if False:
            while True:
                i = 10
        return self.server.get_session(session_id)

    def _attach_session(self, session_id, start_heartbeat=False):
        if False:
            i = 10
            return i + 15
        session = self._get_session(session_id)
        if session is None:
            session = self.server.create_session(session_id)
        if not session.set_handler(self, start_heartbeat):
            return False
        self.session = session
        session.verify_state()
        return True

    def _detach(self):
        if False:
            return 10
        'Detach from the session'
        if self.session:
            self.session.remove_handler(self)
            self.session = None

    def check_xsrf_cookie(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def send_message(self, message, binary=False):
        if False:
            print('Hello World!')
        'Called by the session when some data is available'
        raise NotImplementedError()

    def session_closed(self):
        if False:
            return 10
        'Called by the session when it was closed'
        self._detach()
        self.safe_finish()

    def on_connection_close(self):
        if False:
            i = 10
            return i + 15
        if self.session is not None:
            self.session.close(1002, 'Connection interrupted')
        super(PollingTransportBase, self).on_connection_close()

    def send_complete(self, f=None):
        if False:
            i = 10
            return i + 15
        self._detach()
        if not self._finished:
            self.safe_finish()