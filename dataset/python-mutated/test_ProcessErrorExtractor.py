from ulauncher.modes.extensions.ProcessErrorExtractor import ProcessErrorExtractor

class TestProcessErrorExtractor:

    def test_is_import_error__true(self):
        if False:
            while True:
                i = 10
        e = ProcessErrorExtractor("ModuleNotFoundError: No module named 'mymodule'")
        assert e.is_import_error()

    def test_is_import_error__false(self):
        if False:
            return 10
        e = ProcessErrorExtractor('KeyError: abc')
        assert not e.is_import_error()

    def test_get_missing_package_name__returns_name(self):
        if False:
            return 10
        e = ProcessErrorExtractor("ModuleNotFoundError: No module named 'mymodule'")
        assert e.get_missing_package_name() == 'mymodule'