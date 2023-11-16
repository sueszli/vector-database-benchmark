"""Tests for options manager for :class:`Poly` and public API functions. """
from sympy.polys.polyoptions import Options, Expand, Gens, Wrt, Sort, Order, Field, Greedy, Domain, Split, Gaussian, Extension, Modulus, Symmetric, Strict, Auto, Frac, Formal, Polys, Include, All, Gen, Symbols, Method
from sympy.polys.orderings import lex
from sympy.polys.domains import FF, GF, ZZ, QQ, QQ_I, RR, CC, EX
from sympy.polys.polyerrors import OptionError, GeneratorsError
from sympy.core.numbers import I, Integer
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.testing.pytest import raises
from sympy.abc import x, y, z

def test_Options_clone():
    if False:
        print('Hello World!')
    opt = Options((x, y, z), {'domain': 'ZZ'})
    assert opt.gens == (x, y, z)
    assert opt.domain == ZZ
    assert ('order' in opt) is False
    new_opt = opt.clone({'gens': (x, y), 'order': 'lex'})
    assert opt.gens == (x, y, z)
    assert opt.domain == ZZ
    assert ('order' in opt) is False
    assert new_opt.gens == (x, y)
    assert new_opt.domain == ZZ
    assert ('order' in new_opt) is True

def test_Expand_preprocess():
    if False:
        print('Hello World!')
    assert Expand.preprocess(False) is False
    assert Expand.preprocess(True) is True
    assert Expand.preprocess(0) is False
    assert Expand.preprocess(1) is True
    raises(OptionError, lambda : Expand.preprocess(x))

def test_Expand_postprocess():
    if False:
        return 10
    opt = {'expand': True}
    Expand.postprocess(opt)
    assert opt == {'expand': True}

def test_Gens_preprocess():
    if False:
        i = 10
        return i + 15
    assert Gens.preprocess((None,)) == ()
    assert Gens.preprocess((x, y, z)) == (x, y, z)
    assert Gens.preprocess(((x, y, z),)) == (x, y, z)
    a = Symbol('a', commutative=False)
    raises(GeneratorsError, lambda : Gens.preprocess((x, x, y)))
    raises(GeneratorsError, lambda : Gens.preprocess((x, y, a)))

def test_Gens_postprocess():
    if False:
        print('Hello World!')
    opt = {'gens': (x, y)}
    Gens.postprocess(opt)
    assert opt == {'gens': (x, y)}

def test_Wrt_preprocess():
    if False:
        print('Hello World!')
    assert Wrt.preprocess(x) == ['x']
    assert Wrt.preprocess('') == []
    assert Wrt.preprocess(' ') == []
    assert Wrt.preprocess('x,y') == ['x', 'y']
    assert Wrt.preprocess('x y') == ['x', 'y']
    assert Wrt.preprocess('x, y') == ['x', 'y']
    assert Wrt.preprocess('x , y') == ['x', 'y']
    assert Wrt.preprocess(' x, y') == ['x', 'y']
    assert Wrt.preprocess(' x,  y') == ['x', 'y']
    assert Wrt.preprocess([x, y]) == ['x', 'y']
    raises(OptionError, lambda : Wrt.preprocess(','))
    raises(OptionError, lambda : Wrt.preprocess(0))

def test_Wrt_postprocess():
    if False:
        return 10
    opt = {'wrt': ['x']}
    Wrt.postprocess(opt)
    assert opt == {'wrt': ['x']}

def test_Sort_preprocess():
    if False:
        i = 10
        return i + 15
    assert Sort.preprocess([x, y, z]) == ['x', 'y', 'z']
    assert Sort.preprocess((x, y, z)) == ['x', 'y', 'z']
    assert Sort.preprocess('x > y > z') == ['x', 'y', 'z']
    assert Sort.preprocess('x>y>z') == ['x', 'y', 'z']
    raises(OptionError, lambda : Sort.preprocess(0))
    raises(OptionError, lambda : Sort.preprocess({x, y, z}))

def test_Sort_postprocess():
    if False:
        while True:
            i = 10
    opt = {'sort': 'x > y'}
    Sort.postprocess(opt)
    assert opt == {'sort': 'x > y'}

def test_Order_preprocess():
    if False:
        print('Hello World!')
    assert Order.preprocess('lex') == lex

def test_Order_postprocess():
    if False:
        while True:
            i = 10
    opt = {'order': True}
    Order.postprocess(opt)
    assert opt == {'order': True}

def test_Field_preprocess():
    if False:
        return 10
    assert Field.preprocess(False) is False
    assert Field.preprocess(True) is True
    assert Field.preprocess(0) is False
    assert Field.preprocess(1) is True
    raises(OptionError, lambda : Field.preprocess(x))

def test_Field_postprocess():
    if False:
        print('Hello World!')
    opt = {'field': True}
    Field.postprocess(opt)
    assert opt == {'field': True}

def test_Greedy_preprocess():
    if False:
        for i in range(10):
            print('nop')
    assert Greedy.preprocess(False) is False
    assert Greedy.preprocess(True) is True
    assert Greedy.preprocess(0) is False
    assert Greedy.preprocess(1) is True
    raises(OptionError, lambda : Greedy.preprocess(x))

def test_Greedy_postprocess():
    if False:
        print('Hello World!')
    opt = {'greedy': True}
    Greedy.postprocess(opt)
    assert opt == {'greedy': True}

def test_Domain_preprocess():
    if False:
        return 10
    assert Domain.preprocess(ZZ) == ZZ
    assert Domain.preprocess(QQ) == QQ
    assert Domain.preprocess(EX) == EX
    assert Domain.preprocess(FF(2)) == FF(2)
    assert Domain.preprocess(ZZ[x, y]) == ZZ[x, y]
    assert Domain.preprocess('Z') == ZZ
    assert Domain.preprocess('Q') == QQ
    assert Domain.preprocess('ZZ') == ZZ
    assert Domain.preprocess('QQ') == QQ
    assert Domain.preprocess('EX') == EX
    assert Domain.preprocess('FF(23)') == FF(23)
    assert Domain.preprocess('GF(23)') == GF(23)
    raises(OptionError, lambda : Domain.preprocess('Z[]'))
    assert Domain.preprocess('Z[x]') == ZZ[x]
    assert Domain.preprocess('Q[x]') == QQ[x]
    assert Domain.preprocess('R[x]') == RR[x]
    assert Domain.preprocess('C[x]') == CC[x]
    assert Domain.preprocess('ZZ[x]') == ZZ[x]
    assert Domain.preprocess('QQ[x]') == QQ[x]
    assert Domain.preprocess('RR[x]') == RR[x]
    assert Domain.preprocess('CC[x]') == CC[x]
    assert Domain.preprocess('Z[x,y]') == ZZ[x, y]
    assert Domain.preprocess('Q[x,y]') == QQ[x, y]
    assert Domain.preprocess('R[x,y]') == RR[x, y]
    assert Domain.preprocess('C[x,y]') == CC[x, y]
    assert Domain.preprocess('ZZ[x,y]') == ZZ[x, y]
    assert Domain.preprocess('QQ[x,y]') == QQ[x, y]
    assert Domain.preprocess('RR[x,y]') == RR[x, y]
    assert Domain.preprocess('CC[x,y]') == CC[x, y]
    raises(OptionError, lambda : Domain.preprocess('Z()'))
    assert Domain.preprocess('Z(x)') == ZZ.frac_field(x)
    assert Domain.preprocess('Q(x)') == QQ.frac_field(x)
    assert Domain.preprocess('ZZ(x)') == ZZ.frac_field(x)
    assert Domain.preprocess('QQ(x)') == QQ.frac_field(x)
    assert Domain.preprocess('Z(x,y)') == ZZ.frac_field(x, y)
    assert Domain.preprocess('Q(x,y)') == QQ.frac_field(x, y)
    assert Domain.preprocess('ZZ(x,y)') == ZZ.frac_field(x, y)
    assert Domain.preprocess('QQ(x,y)') == QQ.frac_field(x, y)
    assert Domain.preprocess('Q<I>') == QQ.algebraic_field(I)
    assert Domain.preprocess('QQ<I>') == QQ.algebraic_field(I)
    assert Domain.preprocess('Q<sqrt(2), I>') == QQ.algebraic_field(sqrt(2), I)
    assert Domain.preprocess('QQ<sqrt(2), I>') == QQ.algebraic_field(sqrt(2), I)
    raises(OptionError, lambda : Domain.preprocess('abc'))

def test_Domain_postprocess():
    if False:
        i = 10
        return i + 15
    raises(GeneratorsError, lambda : Domain.postprocess({'gens': (x, y), 'domain': ZZ[y, z]}))
    raises(GeneratorsError, lambda : Domain.postprocess({'gens': (), 'domain': EX}))
    raises(GeneratorsError, lambda : Domain.postprocess({'domain': EX}))

def test_Split_preprocess():
    if False:
        print('Hello World!')
    assert Split.preprocess(False) is False
    assert Split.preprocess(True) is True
    assert Split.preprocess(0) is False
    assert Split.preprocess(1) is True
    raises(OptionError, lambda : Split.preprocess(x))

def test_Split_postprocess():
    if False:
        return 10
    raises(NotImplementedError, lambda : Split.postprocess({'split': True}))

def test_Gaussian_preprocess():
    if False:
        i = 10
        return i + 15
    assert Gaussian.preprocess(False) is False
    assert Gaussian.preprocess(True) is True
    assert Gaussian.preprocess(0) is False
    assert Gaussian.preprocess(1) is True
    raises(OptionError, lambda : Gaussian.preprocess(x))

def test_Gaussian_postprocess():
    if False:
        while True:
            i = 10
    opt = {'gaussian': True}
    Gaussian.postprocess(opt)
    assert opt == {'gaussian': True, 'domain': QQ_I}

def test_Extension_preprocess():
    if False:
        i = 10
        return i + 15
    assert Extension.preprocess(True) is True
    assert Extension.preprocess(1) is True
    assert Extension.preprocess([]) is None
    assert Extension.preprocess(sqrt(2)) == {sqrt(2)}
    assert Extension.preprocess([sqrt(2)]) == {sqrt(2)}
    assert Extension.preprocess([sqrt(2), I]) == {sqrt(2), I}
    raises(OptionError, lambda : Extension.preprocess(False))
    raises(OptionError, lambda : Extension.preprocess(0))

def test_Extension_postprocess():
    if False:
        print('Hello World!')
    opt = {'extension': {sqrt(2)}}
    Extension.postprocess(opt)
    assert opt == {'extension': {sqrt(2)}, 'domain': QQ.algebraic_field(sqrt(2))}
    opt = {'extension': True}
    Extension.postprocess(opt)
    assert opt == {'extension': True}

def test_Modulus_preprocess():
    if False:
        for i in range(10):
            print('nop')
    assert Modulus.preprocess(23) == 23
    assert Modulus.preprocess(Integer(23)) == 23
    raises(OptionError, lambda : Modulus.preprocess(0))
    raises(OptionError, lambda : Modulus.preprocess(x))

def test_Modulus_postprocess():
    if False:
        i = 10
        return i + 15
    opt = {'modulus': 5}
    Modulus.postprocess(opt)
    assert opt == {'modulus': 5, 'domain': FF(5)}
    opt = {'modulus': 5, 'symmetric': False}
    Modulus.postprocess(opt)
    assert opt == {'modulus': 5, 'domain': FF(5, False), 'symmetric': False}

def test_Symmetric_preprocess():
    if False:
        while True:
            i = 10
    assert Symmetric.preprocess(False) is False
    assert Symmetric.preprocess(True) is True
    assert Symmetric.preprocess(0) is False
    assert Symmetric.preprocess(1) is True
    raises(OptionError, lambda : Symmetric.preprocess(x))

def test_Symmetric_postprocess():
    if False:
        for i in range(10):
            print('nop')
    opt = {'symmetric': True}
    Symmetric.postprocess(opt)
    assert opt == {'symmetric': True}

def test_Strict_preprocess():
    if False:
        print('Hello World!')
    assert Strict.preprocess(False) is False
    assert Strict.preprocess(True) is True
    assert Strict.preprocess(0) is False
    assert Strict.preprocess(1) is True
    raises(OptionError, lambda : Strict.preprocess(x))

def test_Strict_postprocess():
    if False:
        return 10
    opt = {'strict': True}
    Strict.postprocess(opt)
    assert opt == {'strict': True}

def test_Auto_preprocess():
    if False:
        i = 10
        return i + 15
    assert Auto.preprocess(False) is False
    assert Auto.preprocess(True) is True
    assert Auto.preprocess(0) is False
    assert Auto.preprocess(1) is True
    raises(OptionError, lambda : Auto.preprocess(x))

def test_Auto_postprocess():
    if False:
        i = 10
        return i + 15
    opt = {'auto': True}
    Auto.postprocess(opt)
    assert opt == {'auto': True}

def test_Frac_preprocess():
    if False:
        while True:
            i = 10
    assert Frac.preprocess(False) is False
    assert Frac.preprocess(True) is True
    assert Frac.preprocess(0) is False
    assert Frac.preprocess(1) is True
    raises(OptionError, lambda : Frac.preprocess(x))

def test_Frac_postprocess():
    if False:
        i = 10
        return i + 15
    opt = {'frac': True}
    Frac.postprocess(opt)
    assert opt == {'frac': True}

def test_Formal_preprocess():
    if False:
        for i in range(10):
            print('nop')
    assert Formal.preprocess(False) is False
    assert Formal.preprocess(True) is True
    assert Formal.preprocess(0) is False
    assert Formal.preprocess(1) is True
    raises(OptionError, lambda : Formal.preprocess(x))

def test_Formal_postprocess():
    if False:
        i = 10
        return i + 15
    opt = {'formal': True}
    Formal.postprocess(opt)
    assert opt == {'formal': True}

def test_Polys_preprocess():
    if False:
        for i in range(10):
            print('nop')
    assert Polys.preprocess(False) is False
    assert Polys.preprocess(True) is True
    assert Polys.preprocess(0) is False
    assert Polys.preprocess(1) is True
    raises(OptionError, lambda : Polys.preprocess(x))

def test_Polys_postprocess():
    if False:
        i = 10
        return i + 15
    opt = {'polys': True}
    Polys.postprocess(opt)
    assert opt == {'polys': True}

def test_Include_preprocess():
    if False:
        for i in range(10):
            print('nop')
    assert Include.preprocess(False) is False
    assert Include.preprocess(True) is True
    assert Include.preprocess(0) is False
    assert Include.preprocess(1) is True
    raises(OptionError, lambda : Include.preprocess(x))

def test_Include_postprocess():
    if False:
        for i in range(10):
            print('nop')
    opt = {'include': True}
    Include.postprocess(opt)
    assert opt == {'include': True}

def test_All_preprocess():
    if False:
        print('Hello World!')
    assert All.preprocess(False) is False
    assert All.preprocess(True) is True
    assert All.preprocess(0) is False
    assert All.preprocess(1) is True
    raises(OptionError, lambda : All.preprocess(x))

def test_All_postprocess():
    if False:
        while True:
            i = 10
    opt = {'all': True}
    All.postprocess(opt)
    assert opt == {'all': True}

def test_Gen_postprocess():
    if False:
        i = 10
        return i + 15
    opt = {'gen': x}
    Gen.postprocess(opt)
    assert opt == {'gen': x}

def test_Symbols_preprocess():
    if False:
        return 10
    raises(OptionError, lambda : Symbols.preprocess(x))

def test_Symbols_postprocess():
    if False:
        while True:
            i = 10
    opt = {'symbols': [x, y, z]}
    Symbols.postprocess(opt)
    assert opt == {'symbols': [x, y, z]}

def test_Method_preprocess():
    if False:
        return 10
    raises(OptionError, lambda : Method.preprocess(10))

def test_Method_postprocess():
    if False:
        while True:
            i = 10
    opt = {'method': 'f5b'}
    Method.postprocess(opt)
    assert opt == {'method': 'f5b'}