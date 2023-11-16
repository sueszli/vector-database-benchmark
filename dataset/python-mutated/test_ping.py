"""External tests of 'bzr ping'"""
import bzrlib
from bzrlib import tests

class TestSmartServerPing(tests.TestCaseWithTransport):

    def test_simple_ping(self):
        if False:
            return 10
        self.setup_smart_server_with_call_log()
        t = self.make_branch_and_tree('branch')
        self.build_tree_contents([('branch/foo', 'thecontents')])
        t.add('foo')
        t.commit('message')
        self.reset_smart_call_log()
        (out, err) = self.run_bzr(['ping', self.get_url('branch')])
        self.assertLength(1, self.hpss_calls)
        self.assertLength(1, self.hpss_connections)
        self.assertEqual(out, "Response: ('ok', '2')\nHeaders: {'Software version': '%s'}\n" % (bzrlib.version_string,))
        self.assertEqual(err, '')