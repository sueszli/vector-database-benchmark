class Base1:

    def __init__(self, *args):
        if False:
            return 10
        print('Base1.__init__', args)

class Clist1(Base1, list):
    pass
a = Clist1()
print(len(a))
print('---')

class Clist2(list, Base1):
    pass