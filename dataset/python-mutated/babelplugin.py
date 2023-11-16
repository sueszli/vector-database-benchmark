"""gettext message extraction via Babel: https://pypi.org/project/Babel/"""
from babel.messages.extract import extract_python
from mako.ext.extract import MessageExtractor

class BabelMakoExtractor(MessageExtractor):

    def __init__(self, keywords, comment_tags, options):
        if False:
            print('Hello World!')
        self.keywords = keywords
        self.options = options
        self.config = {'comment-tags': ' '.join(comment_tags), 'encoding': options.get('input_encoding', options.get('encoding', None))}
        super().__init__()

    def __call__(self, fileobj):
        if False:
            for i in range(10):
                print('nop')
        return self.process_file(fileobj)

    def process_python(self, code, code_lineno, translator_strings):
        if False:
            i = 10
            return i + 15
        comment_tags = self.config['comment-tags']
        for (lineno, funcname, messages, python_translator_comments) in extract_python(code, self.keywords, comment_tags, self.options):
            yield (code_lineno + (lineno - 1), funcname, messages, translator_strings + python_translator_comments)

def extract(fileobj, keywords, comment_tags, options):
    if False:
        i = 10
        return i + 15
    'Extract messages from Mako templates.\n\n    :param fileobj: the file-like object the messages should be extracted from\n    :param keywords: a list of keywords (i.e. function names) that should be\n                     recognized as translation functions\n    :param comment_tags: a list of translator tags to search for and include\n                         in the results\n    :param options: a dictionary of additional options (optional)\n    :return: an iterator over ``(lineno, funcname, message, comments)`` tuples\n    :rtype: ``iterator``\n    '
    extractor = BabelMakoExtractor(keywords, comment_tags, options)
    yield from extractor(fileobj)