class DocFormat:
    """Library to test documentation formatting.

    *bold* or <b>bold</b> http://example.com
    """

    def keyword(self):
        if False:
            i = 10
            return i + 15
        '*bold* or <b>bold</b> http://example.com'

    def link(self):
        if False:
            while True:
                i = 10
        'Link to `Keyword`.'

    def rest(self):
        if False:
            while True:
                i = 10
        "Let's see *how well* reST__ works.\n\n        This documentation is mainly used for manually verifying reST output.\n        This link to \\`Keyword\\` is also automatically tested.\n\n        ====  =====\n        My    table\n        two   rows\n        ====  =====\n\n        - list\n        - here\n\n        Preformatted::\n\n            def example():\n                pass\n\n        __ http://docutils.sourceforge.net\n\n        .. code:: robotframework\n\n            *** Test Cases ***\n            Example\n                Log    How cool is this!?!?!1!\n        "