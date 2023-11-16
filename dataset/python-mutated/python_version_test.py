import testslide
from ..exceptions import InvalidPythonVersion
from ..python_version import PythonVersion

class PythonVersionTest(testslide.TestCase):

    def test_from_string(self) -> None:
        if False:
            i = 10
            return i + 15

        def assert_parsed(input: str, expected: PythonVersion) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(PythonVersion.from_string(input), expected)

        def assert_not_parsed(input: str) -> None:
            if False:
                while True:
                    i = 10
            with self.assertRaises(InvalidPythonVersion):
                PythonVersion.from_string(input)
        assert_not_parsed('')
        assert_not_parsed('derp')
        assert_not_parsed('123abc')
        assert_not_parsed('1.a')
        assert_not_parsed('1.2.a')
        assert_not_parsed('.1')
        assert_not_parsed('1.2.3.4')
        assert_parsed('3', PythonVersion(major=3))
        assert_parsed('3.6', PythonVersion(major=3, minor=6))
        assert_parsed('3.6.7', PythonVersion(major=3, minor=6, micro=7))