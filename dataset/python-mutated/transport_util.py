from bzrlib.tests import features
if features.paramiko.available():
    from bzrlib.tests import test_sftp_transport
    from bzrlib.transport import sftp, Transport
    _backing_scheme = 'sftp'
    _backing_transport_class = sftp.SFTPTransport
    _backing_test_class = test_sftp_transport.TestCaseWithSFTPServer
else:
    from bzrlib.transport import ftp, Transport
    from bzrlib.tests import test_ftp_transport
    _backing_scheme = 'ftp'
    _backing_transport_class = ftp.FtpTransport
    _backing_test_class = test_ftp_transport.TestCaseWithFTPServer

class TestCaseWithConnectionHookedTransport(_backing_test_class):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestCaseWithConnectionHookedTransport, self).setUp()
        self.reset_connections()

    def start_logging_connections(self):
        if False:
            i = 10
            return i + 15
        Transport.hooks.install_named_hook('post_connect', self.connections.append, None)

    def reset_connections(self):
        if False:
            i = 10
            return i + 15
        self.connections = []