from twisted.spread import pb

class QuoteReader(pb.Root):

    def __init__(self, quoter):
        if False:
            return 10
        self.quoter = quoter

    def remote_nextQuote(self):
        if False:
            for i in range(10):
                print('nop')
        return self.quoter.getQuote()