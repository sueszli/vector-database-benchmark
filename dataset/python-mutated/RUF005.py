foo + []
[*foo] + []
first = [1, 2, 3]
second = first + [4, 5, 6]

class Fun:
    words = ('how', 'fun!')

    def yay(self):
        if False:
            while True:
                i = 10
        return self.words
yay = Fun().yay
foo = [4, 5, 6]
bar = [1, 2, 3] + foo
zoob = tuple(bar)
quux = (7, 8, 9) + zoob
spam = quux + (10, 11, 12)
spom = list(spam)
eggs = spom + [13, 14, 15]
elatement = ('we all say',) + yay()
excitement = ('we all think',) + Fun().yay()
astonishment = ('we all feel',) + Fun.words
chain = ['a', 'b', 'c'] + eggs + list(('yes', 'no', 'pants') + zoob)
baz = () + zoob
[] + foo + []
pylint_call = [sys.executable, '-m', 'pylint'] + args + [path]
pylint_call_tuple = (sys.executable, '-m', 'pylint') + args + (path, path2)
b = a + [2, 3] + [4]
f"{a() + ['b']}"
a = (1,) + [2]
a = [1, 2] + (3, 4)
a = [1, 2, 3] + b + (4, 5, 6)