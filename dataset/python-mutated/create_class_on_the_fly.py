def fn(self, name='world'):
    if False:
        i = 10
        return i + 15
    print('Hello, %s.' % name)
Hello = type('Hello', (object,), dict(hello=fn))
h = Hello()
print('call h.hello():')
h.hello()
print('type(Hello) =', type(Hello))
print('type(h) =', type(h))