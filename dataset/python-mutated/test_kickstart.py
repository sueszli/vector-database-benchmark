import salt.utils.kickstart as kickstart
from tests.support.unit import TestCase

class KickstartTestCase(TestCase):

    def test_clean_args(self):
        if False:
            while True:
                i = 10
        ret = kickstart.clean_args({'foo': 'bar', 'baz': False})
        assert ret == {'foo': 'bar'}, ret