from __future__ import print_function
'\n>>> from args_ext import *\n\n>>> raw(3, 4, foo = \'bar\', baz = 42)\n((3, 4), {\'foo\': \'bar\', \'baz\': 42})\n\n   Prove that we can handle empty keywords and non-keywords\n\n>>> raw(3, 4)\n((3, 4), {})\n\n>>> raw(foo = \'bar\')\n((), {\'foo\': \'bar\'})\n\n>>> f(x= 1, y = 3, z = \'hello\')\n(1, 3.0, \'hello\')\n\n>>> f(z = \'hello\', x = 3, y = 2.5)\n(3, 2.5, \'hello\')\n\n>>> f(1, z = \'hi\', y = 3)\n(1, 3.0, \'hi\')\n\n>>> try: f(1, 2, \'hello\', bar = \'baz\')\n... except TypeError: pass\n... else: print(\'expected an exception: unknown keyword\')\n\n\n   Exercise the functions using default stubs\n\n>>> f1(z = \'nix\', y = .125, x = 2)\n(2, 0.125, \'nix\')\n>>> f1(y = .125, x = 2)\n(2, 0.125, \'wow\')\n>>> f1(x = 2)\n(2, 4.25, \'wow\')\n>>> f1()\n(1, 4.25, \'wow\')\n\n>>> f2(z = \'nix\', y = .125, x = 2)\n(2, 0.125, \'nix\')\n>>> f2(y = .125, x = 2)\n(2, 0.125, \'wow\')\n>>> f2(x = 2)\n(2, 4.25, \'wow\')\n>>> f2()\n(1, 4.25, \'wow\')\n\n>>> f3(z = \'nix\', y = .125, x = 2)\n(2, 0.125, \'nix\')\n>>> f3(y = .125, x = 2)\n(2, 0.125, \'wow\')\n>>> f3(x = 2)\n(2, 4.25, \'wow\')\n>>> f3()\n(1, 4.25, \'wow\')\n\n   Member function tests\n\n>>> q = X()\n>>> q.f(x= 1, y = 3, z = \'hello\')\n(1, 3.0, \'hello\')\n\n>>> q.f(z = \'hello\', x = 3, y = 2.5)\n(3, 2.5, \'hello\')\n\n>>> q.f(1, z = \'hi\', y = 3)\n(1, 3.0, \'hi\')\n\n>>> try: q.f(1, 2, \'hello\', bar = \'baz\')\n... except TypeError: pass\n... else: print(\'expected an exception: unknown keyword\')\n\n   Exercise member functions using default stubs\n\n>>> q.f1(z = \'nix\', y = .125, x = 2)\n(2, 0.125, \'nix\')\n>>> q.f1(y = .125, x = 2)\n(2, 0.125, \'wow\')\n>>> q.f1(x = 2)\n(2, 4.25, \'wow\')\n>>> q.f1()\n(1, 4.25, \'wow\')\n>>> q.f2.__doc__.splitlines()[1]\n\'f2( (X)self [, (int)x [, (float)y [, (str)z]]]) -> tuple :\'\n\n>>> q.f2.__doc__.splitlines()[2]\n"    f2\'s docstring"\n\n>>> X.f.__doc__.splitlines()[1:5]\n[\'f( (X)self, (int)x, (float)y, (str)z) -> tuple :\', "    This is X.f\'s docstring", \'\', \'    C++ signature :\']\n\n>>> xfuncs = (X.inner0, X.inner1, X.inner2, X.inner3, X.inner4, X.inner5)\n>>> for f in xfuncs:\n...    print(f(q,1).value(), end=\' \')\n...    print(f(q, n = 1).value(), end=\' \')\n...    print(f(q, n = 0).value(), end=\' \')\n...    print(f.__doc__.splitlines()[1:5])\n1 1 0 [\'inner0( (X)self, (bool)n) -> Y :\', \'    docstring\', \'\', \'    C++ signature :\']\n1 1 0 [\'inner1( (X)self, (bool)n) -> Y :\', \'    docstring\', \'\', \'    C++ signature :\']\n1 1 0 [\'inner2( (X)self, (bool)n) -> Y :\', \'    docstring\', \'\', \'    C++ signature :\']\n1 1 0 [\'inner3( (X)self, (bool)n) -> Y :\', \'    docstring\', \'\', \'    C++ signature :\']\n1 1 0 [\'inner4( (X)self, (bool)n) -> Y :\', \'    docstring\', \'\', \'    C++ signature :\']\n1 1 0 [\'inner5( (X)self, (bool)n) -> Y :\', \'    docstring\', \'\', \'    C++ signature :\']\n\n>>> x = X(a1 = 44, a0 = 22)\n>>> x.inner0(0).value()\n22\n>>> x.inner0(1).value()\n44\n\n>>> x = X(a0 = 7)\n>>> x.inner0(0).value()\n7\n>>> x.inner0(1).value()\n1\n\n>>> inner(n = 1, self = q).value()\n1\n\n>>> y = Y(value = 33)\n>>> y.raw(this = 1, that = \'the other\')[1]\n{\'this\': 1, \'that\': \'the other\'}\n\n'

def run(args=None):
    if False:
        return 10
    import sys
    import doctest
    if args is not None:
        sys.argv = args
    return doctest.testmod(sys.modules.get(__name__))
if __name__ == '__main__':
    print('running...')
    import sys
    status = run()[0]
    if status == 0:
        print('Done.')
    import args_ext
    help(args_ext)
    sys.exit(status)