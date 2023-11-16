from sympy.core import symbols
from sympy.crypto.crypto import cycle_list, encipher_shift, encipher_affine, encipher_substitution, check_and_join, encipher_vigenere, decipher_vigenere, encipher_hill, decipher_hill, encipher_bifid5, encipher_bifid6, bifid5_square, bifid6_square, bifid5, bifid6, decipher_bifid5, decipher_bifid6, encipher_kid_rsa, decipher_kid_rsa, kid_rsa_private_key, kid_rsa_public_key, decipher_rsa, rsa_private_key, rsa_public_key, encipher_rsa, lfsr_connection_polynomial, lfsr_autocorrelation, lfsr_sequence, encode_morse, decode_morse, elgamal_private_key, elgamal_public_key, encipher_elgamal, decipher_elgamal, dh_private_key, dh_public_key, dh_shared_key, decipher_shift, decipher_affine, encipher_bifid, decipher_bifid, bifid_square, padded_key, uniq, decipher_gm, encipher_gm, gm_public_key, gm_private_key, encipher_bg, decipher_bg, bg_private_key, bg_public_key, encipher_rot13, decipher_rot13, encipher_atbash, decipher_atbash, NonInvertibleCipherWarning, encipher_railfence, decipher_railfence
from sympy.external.gmpy import gcd
from sympy.matrices import Matrix
from sympy.ntheory import isprime, is_primitive_root
from sympy.polys.domains import FF
from sympy.testing.pytest import raises, warns
from sympy.core.random import randrange

def test_encipher_railfence():
    if False:
        return 10
    assert encipher_railfence('hello world', 2) == 'hlowrdel ol'
    assert encipher_railfence('hello world', 3) == 'horel ollwd'
    assert encipher_railfence('hello world', 4) == 'hwe olordll'

def test_decipher_railfence():
    if False:
        return 10
    assert decipher_railfence('hlowrdel ol', 2) == 'hello world'
    assert decipher_railfence('horel ollwd', 3) == 'hello world'
    assert decipher_railfence('hwe olordll', 4) == 'hello world'

def test_cycle_list():
    if False:
        for i in range(10):
            print('nop')
    assert cycle_list(3, 4) == [3, 0, 1, 2]
    assert cycle_list(-1, 4) == [3, 0, 1, 2]
    assert cycle_list(1, 4) == [1, 2, 3, 0]

def test_encipher_shift():
    if False:
        for i in range(10):
            print('nop')
    assert encipher_shift('ABC', 0) == 'ABC'
    assert encipher_shift('ABC', 1) == 'BCD'
    assert encipher_shift('ABC', -1) == 'ZAB'
    assert decipher_shift('ZAB', -1) == 'ABC'

def test_encipher_rot13():
    if False:
        return 10
    assert encipher_rot13('ABC') == 'NOP'
    assert encipher_rot13('NOP') == 'ABC'
    assert decipher_rot13('ABC') == 'NOP'
    assert decipher_rot13('NOP') == 'ABC'

def test_encipher_affine():
    if False:
        print('Hello World!')
    assert encipher_affine('ABC', (1, 0)) == 'ABC'
    assert encipher_affine('ABC', (1, 1)) == 'BCD'
    assert encipher_affine('ABC', (-1, 0)) == 'AZY'
    assert encipher_affine('ABC', (-1, 1), symbols='ABCD') == 'BAD'
    assert encipher_affine('123', (-1, 1), symbols='1234') == '214'
    assert encipher_affine('ABC', (3, 16)) == 'QTW'
    assert decipher_affine('QTW', (3, 16)) == 'ABC'

def test_encipher_atbash():
    if False:
        print('Hello World!')
    assert encipher_atbash('ABC') == 'ZYX'
    assert encipher_atbash('ZYX') == 'ABC'
    assert decipher_atbash('ABC') == 'ZYX'
    assert decipher_atbash('ZYX') == 'ABC'

def test_encipher_substitution():
    if False:
        while True:
            i = 10
    assert encipher_substitution('ABC', 'BAC', 'ABC') == 'BAC'
    assert encipher_substitution('123', '1243', '1234') == '124'

def test_check_and_join():
    if False:
        return 10
    assert check_and_join('abc') == 'abc'
    assert check_and_join(uniq('aaabc')) == 'abc'
    assert check_and_join('ab c'.split()) == 'abc'
    assert check_and_join('abc', 'a', filter=True) == 'a'
    raises(ValueError, lambda : check_and_join('ab', 'a'))

def test_encipher_vigenere():
    if False:
        for i in range(10):
            print('nop')
    assert encipher_vigenere('ABC', 'ABC') == 'ACE'
    assert encipher_vigenere('ABC', 'ABC', symbols='ABCD') == 'ACA'
    assert encipher_vigenere('ABC', 'AB', symbols='ABCD') == 'ACC'
    assert encipher_vigenere('AB', 'ABC', symbols='ABCD') == 'AC'
    assert encipher_vigenere('A', 'ABC', symbols='ABCD') == 'A'

def test_decipher_vigenere():
    if False:
        while True:
            i = 10
    assert decipher_vigenere('ABC', 'ABC') == 'AAA'
    assert decipher_vigenere('ABC', 'ABC', symbols='ABCD') == 'AAA'
    assert decipher_vigenere('ABC', 'AB', symbols='ABCD') == 'AAC'
    assert decipher_vigenere('AB', 'ABC', symbols='ABCD') == 'AA'
    assert decipher_vigenere('A', 'ABC', symbols='ABCD') == 'A'

def test_encipher_hill():
    if False:
        for i in range(10):
            print('nop')
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert encipher_hill('ABCD', A) == 'CFIV'
    A = Matrix(2, 2, [1, 0, 0, 1])
    assert encipher_hill('ABCD', A) == 'ABCD'
    assert encipher_hill('ABCD', A, symbols='ABCD') == 'ABCD'
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert encipher_hill('ABCD', A, symbols='ABCD') == 'CBAB'
    assert encipher_hill('AB', A, symbols='ABCD') == 'CB'
    assert encipher_hill('ABA', A) == 'CFGC'
    assert encipher_hill('ABA', A, pad='Z') == 'CFYV'

def test_decipher_hill():
    if False:
        return 10
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert decipher_hill('CFIV', A) == 'ABCD'
    A = Matrix(2, 2, [1, 0, 0, 1])
    assert decipher_hill('ABCD', A) == 'ABCD'
    assert decipher_hill('ABCD', A, symbols='ABCD') == 'ABCD'
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert decipher_hill('CBAB', A, symbols='ABCD') == 'ABCD'
    assert decipher_hill('CB', A, symbols='ABCD') == 'AB'
    assert decipher_hill('CFA', A) == 'ABAA'

def test_encipher_bifid5():
    if False:
        for i in range(10):
            print('nop')
    assert encipher_bifid5('AB', 'AB') == 'AB'
    assert encipher_bifid5('AB', 'CD') == 'CO'
    assert encipher_bifid5('ab', 'c') == 'CH'
    assert encipher_bifid5('a bc', 'b') == 'BAC'

def test_bifid5_square():
    if False:
        return 10
    A = bifid5
    f = lambda i, j: symbols(A[5 * i + j])
    M = Matrix(5, 5, f)
    assert bifid5_square('') == M

def test_decipher_bifid5():
    if False:
        i = 10
        return i + 15
    assert decipher_bifid5('AB', 'AB') == 'AB'
    assert decipher_bifid5('CO', 'CD') == 'AB'
    assert decipher_bifid5('ch', 'c') == 'AB'
    assert decipher_bifid5('b ac', 'b') == 'ABC'

def test_encipher_bifid6():
    if False:
        print('Hello World!')
    assert encipher_bifid6('AB', 'AB') == 'AB'
    assert encipher_bifid6('AB', 'CD') == 'CP'
    assert encipher_bifid6('ab', 'c') == 'CI'
    assert encipher_bifid6('a bc', 'b') == 'BAC'

def test_decipher_bifid6():
    if False:
        while True:
            i = 10
    assert decipher_bifid6('AB', 'AB') == 'AB'
    assert decipher_bifid6('CP', 'CD') == 'AB'
    assert decipher_bifid6('ci', 'c') == 'AB'
    assert decipher_bifid6('b ac', 'b') == 'ABC'

def test_bifid6_square():
    if False:
        i = 10
        return i + 15
    A = bifid6
    f = lambda i, j: symbols(A[6 * i + j])
    M = Matrix(6, 6, f)
    assert bifid6_square('') == M

def test_rsa_public_key():
    if False:
        print('Hello World!')
    assert rsa_public_key(2, 3, 1) == (6, 1)
    assert rsa_public_key(5, 3, 3) == (15, 3)
    with warns(NonInvertibleCipherWarning):
        assert rsa_public_key(2, 2, 1) == (4, 1)
        assert rsa_public_key(8, 8, 8) is False

def test_rsa_private_key():
    if False:
        for i in range(10):
            print('nop')
    assert rsa_private_key(2, 3, 1) == (6, 1)
    assert rsa_private_key(5, 3, 3) == (15, 3)
    assert rsa_private_key(23, 29, 5) == (667, 493)
    with warns(NonInvertibleCipherWarning):
        assert rsa_private_key(2, 2, 1) == (4, 1)
        assert rsa_private_key(8, 8, 8) is False

def test_rsa_large_key():
    if False:
        print('Hello World!')
    p = int('101565610013301240713207239558950144682174355406589305284428666903702505233009')
    q = int('89468719188754548893545560595594841381237600305314352142924213312069293984003')
    e = int('65537')
    d = int('8936505818327042395303988587447591295947962354408444794561435666999402846577625762582824202269399672579058991442587406384754958587400493169361356902030209')
    assert rsa_public_key(p, q, e) == (p * q, e)
    assert rsa_private_key(p, q, e) == (p * q, d)

def test_encipher_rsa():
    if False:
        i = 10
        return i + 15
    puk = rsa_public_key(2, 3, 1)
    assert encipher_rsa(2, puk) == 2
    puk = rsa_public_key(5, 3, 3)
    assert encipher_rsa(2, puk) == 8
    with warns(NonInvertibleCipherWarning):
        puk = rsa_public_key(2, 2, 1)
        assert encipher_rsa(2, puk) == 2

def test_decipher_rsa():
    if False:
        print('Hello World!')
    prk = rsa_private_key(2, 3, 1)
    assert decipher_rsa(2, prk) == 2
    prk = rsa_private_key(5, 3, 3)
    assert decipher_rsa(8, prk) == 2
    with warns(NonInvertibleCipherWarning):
        prk = rsa_private_key(2, 2, 1)
        assert decipher_rsa(2, prk) == 2

def test_mutltiprime_rsa_full_example():
    if False:
        return 10
    puk = rsa_public_key(2, 3, 5, 7, 11, 13, 7)
    prk = rsa_private_key(2, 3, 5, 7, 11, 13, 7)
    assert puk == (30030, 7)
    assert prk == (30030, 823)
    msg = 10
    encrypted = encipher_rsa(2 * msg - 15, puk)
    assert encrypted == 18065
    decrypted = (decipher_rsa(encrypted, prk) + 15) / 2
    assert decrypted == msg
    puk1 = rsa_public_key(53, 41, 43, 47, 41)
    prk1 = rsa_private_key(53, 41, 43, 47, 41)
    puk2 = rsa_public_key(53, 41, 43, 47, 97)
    prk2 = rsa_private_key(53, 41, 43, 47, 97)
    assert puk1 == (4391633, 41)
    assert prk1 == (4391633, 294041)
    assert puk2 == (4391633, 97)
    assert prk2 == (4391633, 455713)
    msg = 12321
    encrypted = encipher_rsa(encipher_rsa(msg, puk1), puk2)
    assert encrypted == 1081588
    decrypted = decipher_rsa(decipher_rsa(encrypted, prk2), prk1)
    assert decrypted == msg

def test_rsa_crt_extreme():
    if False:
        return 10
    p = int('101771576071542450680238615036930821209064871437250622834065015408225822620404699983829716714082136463818069719487950024555765445186962893346463841419427008800341257468600224049986260471922572481630144688417254769186394157267097360778136329612909110256421232977833028677441206049309220354796014376698325101693')
    q = int('28752342353095132872290181526607275886182793241660805077850801756895127977542869729522735531281818618305768362897386687452503402819969112887067641411845844290003577887448262476551386164327966696316822188398336199002306588703902894100476186823849595103239410527279605442148285816149368667083114802852804976893')
    r = int('176982292598688257768795007363501868388509619359563101343782618977186218671746306754136969481624522529192113803880017112559607315449521981157084370187887650624061033066022458512942411841187478937899723152771600850861641198795360418753353848448205660287479617671726408053319619892052000850883994343378882717849')
    s = int('68925428438585431029269182233502611027091755064643742383515623643213105828968933955293670749428083531871387944227457184196452829123186515721260426690367759918078989691645612028911275283598502265889669730331688206825220074713977607415178738015831030364290585369150502819743827343552098197095520550865360159439')
    t = int('69035483433453632820551311892368908779778144568711455301541094314870476423226953576968609257479231896350331830698238209105217117290910679774888326149322416241405010692044244589681980660015448444826108008217972129130625571421904893252804729877353352739420480574842850202181462656251626522910618936534699566291')
    e = 65537
    puk = rsa_public_key(p, q, r, s, t, e)
    prk = rsa_private_key(p, q, r, s, t, e)
    plaintext = 1000
    ciphertext_1 = encipher_rsa(plaintext, puk)
    ciphertext_2 = encipher_rsa(plaintext, puk, [p, q, r, s, t])
    assert ciphertext_1 == ciphertext_2
    assert decipher_rsa(ciphertext_1, prk) == decipher_rsa(ciphertext_1, prk, [p, q, r, s, t])

def test_rsa_exhaustive():
    if False:
        i = 10
        return i + 15
    (p, q) = (61, 53)
    e = 17
    puk = rsa_public_key(p, q, e, totient='Carmichael')
    prk = rsa_private_key(p, q, e, totient='Carmichael')
    for msg in range(puk[0]):
        encrypted = encipher_rsa(msg, puk)
        decrypted = decipher_rsa(encrypted, prk)
        try:
            assert decrypted == msg
        except AssertionError:
            raise AssertionError('The RSA is not correctly decrypted (Original : {}, Encrypted : {}, Decrypted : {})'.format(msg, encrypted, decrypted))

def test_rsa_multiprime_exhanstive():
    if False:
        return 10
    primes = [3, 5, 7, 11]
    e = 7
    args = primes + [e]
    puk = rsa_public_key(*args, totient='Carmichael')
    prk = rsa_private_key(*args, totient='Carmichael')
    n = puk[0]
    for msg in range(n):
        encrypted = encipher_rsa(msg, puk)
        decrypted = decipher_rsa(encrypted, prk)
        try:
            assert decrypted == msg
        except AssertionError:
            raise AssertionError('The RSA is not correctly decrypted (Original : {}, Encrypted : {}, Decrypted : {})'.format(msg, encrypted, decrypted))

def test_rsa_multipower_exhanstive():
    if False:
        while True:
            i = 10
    primes = [5, 5, 7]
    e = 7
    args = primes + [e]
    puk = rsa_public_key(*args, multipower=True)
    prk = rsa_private_key(*args, multipower=True)
    n = puk[0]
    for msg in range(n):
        if gcd(msg, n) != 1:
            continue
        encrypted = encipher_rsa(msg, puk)
        decrypted = decipher_rsa(encrypted, prk)
        try:
            assert decrypted == msg
        except AssertionError:
            raise AssertionError('The RSA is not correctly decrypted (Original : {}, Encrypted : {}, Decrypted : {})'.format(msg, encrypted, decrypted))

def test_kid_rsa_public_key():
    if False:
        while True:
            i = 10
    assert kid_rsa_public_key(1, 2, 1, 1) == (5, 2)
    assert kid_rsa_public_key(1, 2, 2, 1) == (8, 3)
    assert kid_rsa_public_key(1, 2, 1, 2) == (7, 2)

def test_kid_rsa_private_key():
    if False:
        i = 10
        return i + 15
    assert kid_rsa_private_key(1, 2, 1, 1) == (5, 3)
    assert kid_rsa_private_key(1, 2, 2, 1) == (8, 3)
    assert kid_rsa_private_key(1, 2, 1, 2) == (7, 4)

def test_encipher_kid_rsa():
    if False:
        i = 10
        return i + 15
    assert encipher_kid_rsa(1, (5, 2)) == 2
    assert encipher_kid_rsa(1, (8, 3)) == 3
    assert encipher_kid_rsa(1, (7, 2)) == 2

def test_decipher_kid_rsa():
    if False:
        return 10
    assert decipher_kid_rsa(2, (5, 3)) == 1
    assert decipher_kid_rsa(3, (8, 3)) == 1
    assert decipher_kid_rsa(2, (7, 4)) == 1

def test_encode_morse():
    if False:
        return 10
    assert encode_morse('ABC') == '.-|-...|-.-.'
    assert encode_morse('SMS ') == '...|--|...||'
    assert encode_morse('SMS\n') == '...|--|...||'
    assert encode_morse('') == ''
    assert encode_morse(' ') == '||'
    assert encode_morse(' ', sep='`') == '``'
    assert encode_morse(' ', sep='``') == '````'
    assert encode_morse('!@#$%^&*()_+') == '-.-.--|.--.-.|...-..-|-.--.|-.--.-|..--.-|.-.-.'
    assert encode_morse('12345') == '.----|..---|...--|....-|.....'
    assert encode_morse('67890') == '-....|--...|---..|----.|-----'

def test_decode_morse():
    if False:
        return 10
    assert decode_morse('-.-|.|-.--') == 'KEY'
    assert decode_morse('.-.|..-|-.||') == 'RUN'
    raises(KeyError, lambda : decode_morse('.....----'))

def test_lfsr_sequence():
    if False:
        for i in range(10):
            print('nop')
    raises(TypeError, lambda : lfsr_sequence(1, [1], 1))
    raises(TypeError, lambda : lfsr_sequence([1], 1, 1))
    F = FF(2)
    assert lfsr_sequence([F(1)], [F(1)], 2) == [F(1), F(1)]
    assert lfsr_sequence([F(0)], [F(1)], 2) == [F(1), F(0)]
    F = FF(3)
    assert lfsr_sequence([F(1)], [F(1)], 2) == [F(1), F(1)]
    assert lfsr_sequence([F(0)], [F(2)], 2) == [F(2), F(0)]
    assert lfsr_sequence([F(1)], [F(2)], 2) == [F(2), F(2)]

def test_lfsr_autocorrelation():
    if False:
        print('Hello World!')
    raises(TypeError, lambda : lfsr_autocorrelation(1, 2, 3))
    F = FF(2)
    s = lfsr_sequence([F(1), F(0)], [F(0), F(1)], 5)
    assert lfsr_autocorrelation(s, 2, 0) == 1
    assert lfsr_autocorrelation(s, 2, 1) == -1

def test_lfsr_connection_polynomial():
    if False:
        i = 10
        return i + 15
    F = FF(2)
    x = symbols('x')
    s = lfsr_sequence([F(1), F(0)], [F(0), F(1)], 5)
    assert lfsr_connection_polynomial(s) == x ** 2 + 1
    s = lfsr_sequence([F(1), F(1)], [F(0), F(1)], 5)
    assert lfsr_connection_polynomial(s) == x ** 2 + x + 1

def test_elgamal_private_key():
    if False:
        print('Hello World!')
    (a, b, _) = elgamal_private_key(digit=100)
    assert isprime(a)
    assert is_primitive_root(b, a)
    assert len(bin(a)) >= 102

def test_elgamal():
    if False:
        for i in range(10):
            print('nop')
    dk = elgamal_private_key(5)
    ek = elgamal_public_key(dk)
    P = ek[0]
    assert P - 1 == decipher_elgamal(encipher_elgamal(P - 1, ek), dk)
    raises(ValueError, lambda : encipher_elgamal(P, dk))
    raises(ValueError, lambda : encipher_elgamal(-1, dk))

def test_dh_private_key():
    if False:
        while True:
            i = 10
    (p, g, _) = dh_private_key(digit=100)
    assert isprime(p)
    assert is_primitive_root(g, p)
    assert len(bin(p)) >= 102

def test_dh_public_key():
    if False:
        return 10
    (p1, g1, a) = dh_private_key(digit=100)
    (p2, g2, ga) = dh_public_key((p1, g1, a))
    assert p1 == p2
    assert g1 == g2
    assert ga == pow(g1, a, p1)

def test_dh_shared_key():
    if False:
        print('Hello World!')
    prk = dh_private_key(digit=100)
    (p, _, ga) = dh_public_key(prk)
    b = randrange(2, p)
    sk = dh_shared_key((p, _, ga), b)
    assert sk == pow(ga, b, p)
    raises(ValueError, lambda : dh_shared_key((1031, 14, 565), 2000))

def test_padded_key():
    if False:
        return 10
    assert padded_key('b', 'ab') == 'ba'
    raises(ValueError, lambda : padded_key('ab', 'ace'))
    raises(ValueError, lambda : padded_key('ab', 'abba'))

def test_bifid():
    if False:
        while True:
            i = 10
    raises(ValueError, lambda : encipher_bifid('abc', 'b', 'abcde'))
    assert encipher_bifid('abc', 'b', 'abcd') == 'bdb'
    raises(ValueError, lambda : decipher_bifid('bdb', 'b', 'abcde'))
    assert encipher_bifid('bdb', 'b', 'abcd') == 'abc'
    raises(ValueError, lambda : bifid_square('abcde'))
    assert bifid5_square('B') == bifid5_square('BACDEFGHIKLMNOPQRSTUVWXYZ')
    assert bifid6_square('B0') == bifid6_square('B0ACDEFGHIJKLMNOPQRSTUVWXYZ123456789')

def test_encipher_decipher_gm():
    if False:
        i = 10
        return i + 15
    ps = [131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    qs = [89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 47]
    messages = [0, 32855, 34303, 14805, 1280, 75859, 38368, 724, 60356, 51675, 76697, 61854, 18661]
    for (p, q) in zip(ps, qs):
        pri = gm_private_key(p, q)
        for msg in messages:
            pub = gm_public_key(p, q)
            enc = encipher_gm(msg, pub)
            dec = decipher_gm(enc, pri)
            assert dec == msg

def test_gm_private_key():
    if False:
        i = 10
        return i + 15
    raises(ValueError, lambda : gm_public_key(13, 15))
    raises(ValueError, lambda : gm_public_key(0, 0))
    raises(ValueError, lambda : gm_public_key(0, 5))
    assert 17, 19 == gm_public_key(17, 19)

def test_gm_public_key():
    if False:
        for i in range(10):
            print('nop')
    assert 323 == gm_public_key(17, 19)[1]
    assert 15 == gm_public_key(3, 5)[1]
    raises(ValueError, lambda : gm_public_key(15, 19))

def test_encipher_decipher_bg():
    if False:
        i = 10
        return i + 15
    ps = [67, 7, 71, 103, 11, 43, 107, 47, 79, 19, 83, 23, 59, 127, 31]
    qs = qs = [7, 71, 103, 11, 43, 107, 47, 79, 19, 83, 23, 59, 127, 31, 67]
    messages = [0, 328, 343, 148, 1280, 758, 383, 724, 603, 516, 766, 618, 186]
    for (p, q) in zip(ps, qs):
        pri = bg_private_key(p, q)
        for msg in messages:
            pub = bg_public_key(p, q)
            enc = encipher_bg(msg, pub)
            dec = decipher_bg(enc, pri)
            assert dec == msg

def test_bg_private_key():
    if False:
        while True:
            i = 10
    raises(ValueError, lambda : bg_private_key(8, 16))
    raises(ValueError, lambda : bg_private_key(8, 8))
    raises(ValueError, lambda : bg_private_key(13, 17))
    assert 23, 31 == bg_private_key(23, 31)

def test_bg_public_key():
    if False:
        while True:
            i = 10
    assert 5293 == bg_public_key(67, 79)
    assert 713 == bg_public_key(23, 31)
    raises(ValueError, lambda : bg_private_key(13, 17))