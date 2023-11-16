from __future__ import absolute_import, division, print_function, unicode_literals
'\n    sockjs.tornado.transports.htmlfile\n    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n    HtmlFile transport implementation.\n'
import re
from octoprint.vendor.sockjs.tornado import proto
from octoprint.vendor.sockjs.tornado.transports import streamingbase
from octoprint.vendor.sockjs.tornado.util import no_auto_finish
try:
    from html import escape
except:
    from cgi import escape
RE = re.compile('[\\W_]+')
HTMLFILE_HEAD = '\n<!doctype html>\n<html><head>\n  <meta http-equiv="X-UA-Compatible" content="IE=edge" />\n  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n</head><body><h2>Don\'t panic!</h2>\n  <script>\n    document.domain = document.domain;\n    var c = parent.%s;\n    c.start();\n    function p(d) {c.message(d);};\n    window.onload = function() {c.stop();};\n  </script>\n'.strip()
HTMLFILE_HEAD += ' ' * (1024 - len(HTMLFILE_HEAD) + 14)
HTMLFILE_HEAD += '\r\n\r\n'

class HtmlFileTransport(streamingbase.StreamingTransportBase):
    name = 'htmlfile'

    def initialize(self, server):
        if False:
            while True:
                i = 10
        super(HtmlFileTransport, self).initialize(server)

    @no_auto_finish
    def get(self, session_id):
        if False:
            i = 10
            return i + 15
        self.preflight()
        self.handle_session_cookie()
        self.disable_cache()
        self.set_header('Content-Type', 'text/html; charset=UTF-8')
        callback = self.get_argument('c', None)
        if not callback:
            self.write('"callback" parameter required')
            self.set_status(500)
            self.finish()
            return
        self.write(HTMLFILE_HEAD % escape(RE.sub('', callback)))
        self.flush()
        if not self._attach_session(session_id):
            self.finish()
            return
        if self.session:
            self.session.flush()

    def send_pack(self, message, binary=False):
        if False:
            while True:
                i = 10
        if binary:
            raise Exception('binary not supported for HtmlFileTransport')
        msg = '<script>\np(%s);\n</script>\r\n' % proto.json_encode(message)
        self.active = False
        try:
            self.notify_sent(len(message))
            self.write(msg)
            self.flush().add_done_callback(self.send_complete)
        except IOError:
            self.session.delayed_close()
            self._detach()