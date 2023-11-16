"""This program is self-checking!"""

class C:

    def sort(self, l, reverse, key_fn):
        if False:
            for i in range(10):
                print('nop')
        return l.sort(reverse=reverse, key=key_fn)

def lcase(s):
    if False:
        print('Hello World!')
    return s.lower()
x = C()
l = ['xyz', 'ABC']
x.sort(l, reverse=False, key_fn=lcase)
assert l == ['ABC', 'xyz']