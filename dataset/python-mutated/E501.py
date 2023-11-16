"""Lorem ipsum dolor sit amet.

    https://github.com/PyCQA/pycodestyle/pull/258/files#diff-841c622497a8033d10152bfdfb15b20b92437ecdea21a260944ea86b77b51533

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
"""
_ = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n\n    https://github.com/PyCQA/pycodestyle/pull/258/files#diff-841c622497a8033d10152bfdfb15b20b92437ecdea21a260944ea86b77b51533\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n'
_ = '---------------------------------------------------------------------------AAAAAAA'
_ = '---------------------------------------------------------------------------亜亜亜亜亜亜亜'

def caller(string: str) -> None:
    if False:
        while True:
            i = 10
    assert string is not None
caller('\n    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n    ')
caller('\n    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n    ')
multiple_strings_per_line = {'Lorem ipsum dolor': 'sit amet', 'Lorem ipsum dolor': 'sit amet consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.', 'Lorem ipsum dolor': '\nsit amet  consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n', 'Lorem ipsum dolor': 'sit amet consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.', 'Lorem ipsum dolor': '\nsit amet  consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n'}
_ = '\nSource: https://github.com/PyCQA/pycodestyle/pull/258/files#diff-841c622497a8033d10152bfdfb15b20b92437ecdea21a260944ea86b77b51533\n'
_ = '\n[this-is-ok](https://github.com/PyCQA/pycodestyle/pull/258/files#diff-841c622497a8033d10152bfdfb15b20b92437ecdea21a260944ea86b77b51533)\n[this is ok](https://github.com/PyCQA/pycodestyle/pull/258/files#diff-841c622497a8033d10152bfdfb15b20b92437ecdea21a260944ea86b77b51533)\n'

class Foo:
    """
    @see https://looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong.url.com

    :param dynamodb_scan_kwargs: kwargs pass to <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html#DynamoDB.Table.scan>
    """

class Bar:
    """
    This is a long sentence that ends with a shortened URL and, therefore, could easily be broken across multiple lines ([source](https://ruff.rs))
    """