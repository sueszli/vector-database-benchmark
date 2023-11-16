from . import Framework

class PublicKey(Framework.TestCase):

    def testAttributes(self):
        if False:
            i = 10
            return i + 15
        self.public_key = self.g.get_user().get_repo('PyGithub').get_public_key()
        self.assertEqual(self.public_key.key, 'u5e1Z25+z8pmgVVt5Pd8k0z/sKpVL1MXYtRAecE4vm8=')
        self.assertEqual(self.public_key.key_id, '568250167242549743')
        self.assertEqual(repr(self.public_key), 'PublicKey(key_id="568250167242549743", key="u5e1Z25+z8pmgVVt5Pd8k0z/sKpVL1MXYtRAecE4vm8=")')

    def testAttributes_with_int_key_id(self):
        if False:
            return 10
        self.public_key = self.g.get_user().get_repo('PyGithub').get_public_key()
        self.assertEqual(self.public_key.key, 'u5e1Z25+z8pmgVVt5Pd8k0z/sKpVL1MXYtRAecE4vm8=')
        self.assertEqual(self.public_key.key_id, 568250167242549743)
        self.assertEqual(repr(self.public_key), 'PublicKey(key_id=568250167242549743, key="u5e1Z25+z8pmgVVt5Pd8k0z/sKpVL1MXYtRAecE4vm8=")')