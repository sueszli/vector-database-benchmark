from ben_scott1_ext import *

class CreatorImpl(Creator):

    def create(self):
        if False:
            print('Hello World!')
        return Product()
factory = Factory()
c = CreatorImpl()
factory.reg(c)
a = factory.create()