from . import Framework

class UserKey(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.key = self.g.get_user().get_key(2626650)

    def testAttributes(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.key.id, 2626650)
        self.assertEqual(self.key.key, 'ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEA2Mm0RjTNAYFfSCtUpO54usdseroUSIYg5KX4JoseTpqyiB/hqewjYLAdUq/tNIQzrkoEJWSyZrQt0ma7/YCyMYuNGd3DU6q6ZAyBeY3E9RyCiKjO3aTL2VKQGFvBVVmGdxGVSCITRphAcsKc/PF35/fg9XP9S0anMXcEFtdfMHz41SSw+XtE+Vc+6cX9FuI5qUfLGbkv8L1v3g4uw9VXlzq4GfTA+1S7D6mcoGHopAIXFlVr+2RfDKdSURMcB22z41fljO1MW4+zUS/4FyUTpL991es5fcwKXYoiE+x06VJeJJ1Krwx+DZj45uweV6cHXt2JwJEI9fWB6WyBlDejWw==')
        self.assertEqual(self.key.title, 'Key added through PyGithub')
        self.assertEqual(self.key.url, 'https://api.github.com/user/keys/2626650')
        self.assertTrue(self.key.verified)
        self.assertEqual(repr(self.key), 'UserKey(title="Key added through PyGithub", id=2626650)')

    def testDelete(self):
        if False:
            print('Hello World!')
        self.key.delete()