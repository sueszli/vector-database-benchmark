class TextToken:

    def __init__(self, text):
        if False:
            for i in range(10):
                print('nop')
        self.text = text

class ArgumentToken:

    def __init__(self, *, name, value, type):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.value = value
        self.type = type

class SearchQueryParser:
    """Simplified and minimal parser for ``name:value`` expressions."""
    allowed_arguments = {'project': list, 'subprojects': list, 'user': str}

    def __init__(self, query):
        if False:
            print('Hello World!')
        self._query = query
        self.query = ''
        self.arguments = {name: type() for (name, type) in self.allowed_arguments.items()}

    def parse(self):
        if False:
            while True:
                i = 10
        "\n        Parse the expression into a query and arguments.\n\n        The parser steps are:\n\n        - Split the string using white spaces.\n        - Tokenize each string into a ``text`` or ``argument`` token.\n          A valid argument has the ``name:value`` form,\n          and it's declared in `allowed_arguments`,\n          anything else is considered a text token.\n        - All text tokens are concatenated to form the final query.\n\n        To interpret an argument as text, it can be escaped as ``name\\:value``.\n        "
        tokens = (self._get_token(text) for text in self._query.split())
        query = []
        for token in tokens:
            if isinstance(token, TextToken):
                query.append(token.text)
            elif isinstance(token, ArgumentToken):
                if token.type == str:
                    self.arguments[token.name] = token.value
                elif token.type == list:
                    self.arguments[token.name].append(token.value)
                else:
                    raise ValueError(f'Invalid argument type {token.type}')
            else:
                raise ValueError('Invalid node')
        self.query = self._unescape(' '.join(query))

    def _get_token(self, text):
        if False:
            for i in range(10):
                print('nop')
        result = text.split(':', maxsplit=1)
        if len(result) < 2:
            return TextToken(text)
        (name, value) = result
        if name in self.allowed_arguments:
            return ArgumentToken(name=name, value=value, type=self.allowed_arguments[name])
        return TextToken(text)

    def _unescape(self, text):
        if False:
            i = 10
            return i + 15
        return text.replace('\\:', ':')