from synapse.api.errors import SynapseError
from synapse.util.stringutils import assert_valid_client_secret, base62_encode
from .. import unittest

class StringUtilsTestCase(unittest.TestCase):

    def test_client_secret_regex(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Ensure that client_secret does not contain illegal characters'
        good = ['abcde12345', 'ABCabc123', '_--something==_', '...--==-18913', '8Dj2odd-e9asd.cd==_--ddas-secret-']
        bad = ['--+-/secret', '\\dx--dsa288', '', 'AAS//', 'asdj**', '>X><Z<!!-)))', 'a@b.com']
        for client_secret in good:
            assert_valid_client_secret(client_secret)
        for client_secret in bad:
            with self.assertRaises(SynapseError):
                assert_valid_client_secret(client_secret)

    def test_base62_encode(self) -> None:
        if False:
            return 10
        self.assertEqual('0', base62_encode(0))
        self.assertEqual('10', base62_encode(62))
        self.assertEqual('1c', base62_encode(100))
        self.assertEqual('001c', base62_encode(100, minwidth=4))