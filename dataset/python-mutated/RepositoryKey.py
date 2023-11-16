from datetime import datetime, timezone
from . import Framework

class RepositoryKey(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.key = self.g.get_user('lra').get_repo('mackup').get_key(21870881)

    def testAttributes(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.key.id, 21870881)
        self.assertEqual(self.key.key, 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDLOoLSVPwG1OSgVSeEXNbfIofYdxR5zs3u4PryhnamfFPYwi2vZW3ZxeI1oRcDh2VEdwGvlN5VUduKJNoOWMVzV2jSyR8CeDHH+I0soQCC7kfJVodU96HcPMzZ6MuVwSfD4BFGvKMXyCnBUqzo28BGHFwVQG8Ya9gL6/cTbuWywgM4xaJgMHv1OVcESXBtBkrqOneTJuOgeEmP0RfUnIAK/3/wbg9mfiBq7JV4cmWAg1xNE8GJoAbci59Tdx1dQgVuuqdQGk5jzNusOVneyMtGEB+p7UpPLJsGBW29rsMt7ITUbXM/kl9v11vPtWb+oOUThoFsDYmsWy7fGGP9YAFB')
        self.assertEqual(self.key.title, 'PyGithub Test Key')
        self.assertEqual(self.key.url, 'https://api.github.com/repos/lra/mackup/keys/21870881')
        self.assertEqual(self.key.created_at, datetime(2017, 2, 22, 8, 16, 23, tzinfo=timezone.utc))
        self.assertTrue(self.key.verified)
        self.assertTrue(self.key.read_only)
        self.assertEqual(repr(self.key), 'RepositoryKey(title="PyGithub Test Key", id=21870881)')

    def testDelete(self):
        if False:
            for i in range(10):
                print('nop')
        self.key.delete()