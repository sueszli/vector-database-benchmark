from random import choice
from zope.interface import implementer
from TwistedQuotes import quoteproto

@implementer(quoteproto.IQuoter)
class StaticQuoter:
    """
    Return a static quote.
    """

    def __init__(self, quote):
        if False:
            for i in range(10):
                print('nop')
        self.quote = quote

    def getQuote(self):
        if False:
            while True:
                i = 10
        return self.quote

@implementer(quoteproto.IQuoter)
class FortuneQuoter:
    """
    Load quotes from a fortune-format file.
    """

    def __init__(self, filenames):
        if False:
            while True:
                i = 10
        self.filenames = filenames

    def getQuote(self):
        if False:
            i = 10
            return i + 15
        with open(choice(self.filenames)) as quoteFile:
            quotes = quoteFile.read().split('\n%\n')
        return choice(quotes)