from numpy.testing import TestCase

class aubiomodule_test_case(TestCase):

    def test_import(self):
        if False:
            while True:
                i = 10
        ' try importing aubio '
        import aubio

    def test_version(self):
        if False:
            i = 10
            return i + 15
        ' test aubio.version '
        import aubio
        self.assertEqual('0', aubio.version[0])
if __name__ == '__main__':
    from unittest import main
    main()