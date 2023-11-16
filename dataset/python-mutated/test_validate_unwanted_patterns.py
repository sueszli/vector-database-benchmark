import io
import pytest
from scripts import validate_unwanted_patterns

class TestBarePytestRaises:

    @pytest.mark.parametrize('data', ['\n    with pytest.raises(ValueError, match="foo"):\n        pass\n    ', '\n    # with pytest.raises(ValueError, match="foo"):\n    #    pass\n    ', '\n    # with pytest.raises(ValueError):\n    #    pass\n    ', '\n    with pytest.raises(\n        ValueError,\n        match="foo"\n    ):\n        pass\n    '])
    def test_pytest_raises(self, data):
        if False:
            for i in range(10):
                print('nop')
        fd = io.StringIO(data.strip())
        result = list(validate_unwanted_patterns.bare_pytest_raises(fd))
        assert result == []

    @pytest.mark.parametrize('data, expected', [('\n    with pytest.raises(ValueError):\n        pass\n    ', [(1, "Bare pytests raise have been found. Please pass in the argument 'match' as well the exception.")]), ('\n    with pytest.raises(ValueError, match="foo"):\n        with pytest.raises(ValueError):\n            pass\n        pass\n    ', [(2, "Bare pytests raise have been found. Please pass in the argument 'match' as well the exception.")]), ('\n    with pytest.raises(ValueError):\n        with pytest.raises(ValueError, match="foo"):\n            pass\n        pass\n    ', [(1, "Bare pytests raise have been found. Please pass in the argument 'match' as well the exception.")]), ('\n    with pytest.raises(\n        ValueError\n    ):\n        pass\n    ', [(1, "Bare pytests raise have been found. Please pass in the argument 'match' as well the exception.")]), ('\n    with pytest.raises(\n        ValueError,\n        # match = "foo"\n    ):\n        pass\n    ', [(1, "Bare pytests raise have been found. Please pass in the argument 'match' as well the exception.")])])
    def test_pytest_raises_raises(self, data, expected):
        if False:
            for i in range(10):
                print('nop')
        fd = io.StringIO(data.strip())
        result = list(validate_unwanted_patterns.bare_pytest_raises(fd))
        assert result == expected

class TestStringsWithWrongPlacedWhitespace:

    @pytest.mark.parametrize('data', ['\n    msg = (\n        "foo\n"\n        " bar"\n    )\n    ', '\n    msg = (\n        "foo"\n        "  bar"\n        "baz"\n    )\n    ', '\n    msg = (\n        f"foo"\n        "  bar"\n    )\n    ', '\n    msg = (\n        "foo"\n        f"  bar"\n    )\n    ', '\n    msg = (\n        "foo"\n        rf"  bar"\n    )\n    '])
    def test_strings_with_wrong_placed_whitespace(self, data):
        if False:
            print('Hello World!')
        fd = io.StringIO(data.strip())
        result = list(validate_unwanted_patterns.strings_with_wrong_placed_whitespace(fd))
        assert result == []

    @pytest.mark.parametrize('data, expected', [('\n    msg = (\n        "foo"\n        " bar"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.')]), ('\n    msg = (\n        f"foo"\n        " bar"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.')]), ('\n    msg = (\n        "foo"\n        f" bar"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.')]), ('\n    msg = (\n        f"foo"\n        f" bar"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.')]), ('\n    msg = (\n        "foo"\n        rf" bar"\n        " baz"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.'), (4, 'String has a space at the beginning instead of the end of the previous string.')]), ('\n    msg = (\n        "foo"\n        " bar"\n        rf" baz"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.'), (4, 'String has a space at the beginning instead of the end of the previous string.')]), ('\n    msg = (\n        "foo"\n        rf" bar"\n        rf" baz"\n    )\n    ', [(3, 'String has a space at the beginning instead of the end of the previous string.'), (4, 'String has a space at the beginning instead of the end of the previous string.')])])
    def test_strings_with_wrong_placed_whitespace_raises(self, data, expected):
        if False:
            for i in range(10):
                print('nop')
        fd = io.StringIO(data.strip())
        result = list(validate_unwanted_patterns.strings_with_wrong_placed_whitespace(fd))
        assert result == expected

class TestNoDefaultUsedNotOnlyForTyping:

    @pytest.mark.parametrize('data', ['\ndef f(\n    a: int | NoDefault,\n    b: float | lib.NoDefault = 0.1,\n    c: pandas._libs.lib.NoDefault = lib.no_default,\n) -> lib.NoDefault | None:\n    pass\n', '\n# var = lib.NoDefault\n# the above is incorrect\na: NoDefault | int\nb: lib.NoDefault = lib.no_default\n'])
    def test_nodefault_used_not_only_for_typing(self, data):
        if False:
            return 10
        fd = io.StringIO(data.strip())
        result = list(validate_unwanted_patterns.nodefault_used_not_only_for_typing(fd))
        assert result == []

    @pytest.mark.parametrize('data, expected', [('\ndef f(\n    a = lib.NoDefault,\n    b: Any\n        = pandas._libs.lib.NoDefault,\n):\n    pass\n', [(2, 'NoDefault is used not only for typing'), (4, 'NoDefault is used not only for typing')]), ('\na: Any = lib.NoDefault\nif a is NoDefault:\n    pass\n', [(1, 'NoDefault is used not only for typing'), (2, 'NoDefault is used not only for typing')])])
    def test_nodefault_used_not_only_for_typing_raises(self, data, expected):
        if False:
            for i in range(10):
                print('nop')
        fd = io.StringIO(data.strip())
        result = list(validate_unwanted_patterns.nodefault_used_not_only_for_typing(fd))
        assert result == expected