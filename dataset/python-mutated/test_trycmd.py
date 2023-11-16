from unittest import mock
from twisted.trial import unittest
from buildbot.clients import tryclient
from buildbot.scripts import trycmd

class TestStatusLog(unittest.TestCase):

    def test_trycmd(self):
        if False:
            i = 10
            return i + 15
        Try = mock.Mock()
        self.patch(tryclient, 'Try', Try)
        inst = Try.return_value = mock.Mock(name='Try-instance')
        rc = trycmd.trycmd({'cfg': 1})
        Try.assert_called_with({'cfg': 1})
        inst.run.assert_called_with()
        self.assertEqual(rc, 0)