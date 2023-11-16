import aubio.cut
from numpy.testing import TestCase

class aubio_cut(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.a_parser = aubio.cut.aubio_cut_parser()

    def test_default_creation(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.a_parser.parse_args(['-v']).verbose
if __name__ == '__main__':
    from unittest import main
    main()