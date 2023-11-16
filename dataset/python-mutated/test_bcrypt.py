import pytest
from pytest_pyodide import run_in_pyodide
_test_vectors = [(b'Kk4DQuMMfZL9o', b'$2b$04$cVWp4XaNU8a4v1uMRum2SO', b'$2b$04$cVWp4XaNU8a4v1uMRum2SO026BWLIoQMD/TXg5uZV.0P.uO8m3YEm'), (b'9IeRXmnGxMYbs', b'$2b$04$pQ7gRO7e6wx/936oXhNjrO', b'$2b$04$pQ7gRO7e6wx/936oXhNjrOUNOHL1D0h1N2IDbJZYs.1ppzSof6SPy'), (b'xVQVbwa1S0M8r', b'$2b$04$SQe9knOzepOVKoYXo9xTte', b'$2b$04$SQe9knOzepOVKoYXo9xTteNYr6MBwVz4tpriJVe3PNgYufGIsgKcW'), (b'Zfgr26LWd22Za', b'$2b$04$eH8zX.q5Q.j2hO1NkVYJQO', b'$2b$04$eH8zX.q5Q.j2hO1NkVYJQOM6KxntS/ow3.YzVmFrE4t//CoF4fvne'), (b'Tg4daC27epFBE', b'$2b$04$ahiTdwRXpUG2JLRcIznxc.', b'$2b$04$ahiTdwRXpUG2JLRcIznxc.s1.ydaPGD372bsGs8NqyYjLY1inG5n2'), (b'xhQPMmwh5ALzW', b'$2b$04$nQn78dV0hGHf5wUBe0zOFu', b'$2b$04$nQn78dV0hGHf5wUBe0zOFu8n07ZbWWOKoGasZKRspZxtt.vBRNMIy'), (b'59je8h5Gj71tg', b'$2b$04$cvXudZ5ugTg95W.rOjMITu', b'$2b$04$cvXudZ5ugTg95W.rOjMITuM1jC0piCl3zF5cmGhzCibHZrNHkmckG'), (b'wT4fHJa2N9WSW', b'$2b$04$YYjtiq4Uh88yUsExO0RNTu', b'$2b$04$YYjtiq4Uh88yUsExO0RNTuEJ.tZlsONac16A8OcLHleWFjVawfGvO'), (b'uSgFRnQdOgm4S', b'$2b$04$WLTjgY/pZSyqX/fbMbJzf.', b'$2b$04$WLTjgY/pZSyqX/fbMbJzf.qxCeTMQOzgL.CimRjMHtMxd/VGKojMu'), (b'tEPtJZXur16Vg', b'$2b$04$2moPs/x/wnCfeQ5pCheMcu', b'$2b$04$2moPs/x/wnCfeQ5pCheMcuSJQ/KYjOZG780UjA/SiR.KsYWNrC7SG'), (b'vvho8C6nlVf9K', b'$2b$04$HrEYC/AQ2HS77G78cQDZQ.', b'$2b$04$HrEYC/AQ2HS77G78cQDZQ.r44WGcruKw03KHlnp71yVQEwpsi3xl2'), (b'5auCCY9by0Ruf', b'$2b$04$vVYgSTfB8KVbmhbZE/k3R.', b'$2b$04$vVYgSTfB8KVbmhbZE/k3R.ux9A0lJUM4CZwCkHI9fifke2.rTF7MG'), (b'GtTkR6qn2QOZW', b'$2b$04$JfoNrR8.doieoI8..F.C1O', b'$2b$04$JfoNrR8.doieoI8..F.C1OQgwE3uTeuardy6lw0AjALUzOARoyf2m'), (b'zKo8vdFSnjX0f', b'$2b$04$HP3I0PUs7KBEzMBNFw7o3O', b'$2b$04$HP3I0PUs7KBEzMBNFw7o3O7f/uxaZU7aaDot1quHMgB2yrwBXsgyy'), (b'I9VfYlacJiwiK', b'$2b$04$xnFVhJsTzsFBTeP3PpgbMe', b'$2b$04$xnFVhJsTzsFBTeP3PpgbMeMREb6rdKV9faW54Sx.yg9plf4jY8qT6'), (b'VFPO7YXnHQbQO', b'$2b$04$WQp9.igoLqVr6Qk70mz6xu', b'$2b$04$WQp9.igoLqVr6Qk70mz6xuRxE0RttVXXdukpR9N54x17ecad34ZF6'), (b'VDx5BdxfxstYk', b'$2b$04$xgZtlonpAHSU/njOCdKztO', b'$2b$04$xgZtlonpAHSU/njOCdKztOPuPFzCNVpB4LGicO4/OGgHv.uKHkwsS'), (b'dEe6XfVGrrfSH', b'$2b$04$2Siw3Nv3Q/gTOIPetAyPr.', b'$2b$04$2Siw3Nv3Q/gTOIPetAyPr.GNj3aO0lb1E5E9UumYGKjP9BYqlNWJe'), (b'cTT0EAFdwJiLn', b'$2b$04$7/Qj7Kd8BcSahPO4khB8me', b'$2b$04$7/Qj7Kd8BcSahPO4khB8me4ssDJCW3r4OGYqPF87jxtrSyPj5cS5m'), (b'J8eHUDuxBB520', b'$2b$04$VvlCUKbTMjaxaYJ.k5juoe', b'$2b$04$VvlCUKbTMjaxaYJ.k5juoecpG/7IzcH1AkmqKi.lIZMVIOLClWAk.'), (b'U*U', b'$2a$05$CCCCCCCCCCCCCCCCCCCCC.', b'$2a$05$CCCCCCCCCCCCCCCCCCCCC.E5YPO9kmyuRGyh0XouQYb4YMJKvyOeW'), (b'U*U*', b'$2a$05$CCCCCCCCCCCCCCCCCCCCC.', b'$2a$05$CCCCCCCCCCCCCCCCCCCCC.VGOzA784oUp/Z0DY336zx7pLYAy0lwK'), (b'U*U*U', b'$2a$05$XXXXXXXXXXXXXXXXXXXXXO', b'$2a$05$XXXXXXXXXXXXXXXXXXXXXOAcXxm9kjPGEMsLznoKqmqw7tc8WCx4a'), (b'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789chars after 72 are ignored', b'$2a$05$abcdefghijklmnopqrstuu', b'$2a$05$abcdefghijklmnopqrstuu5s2v8.iXieOjg/.AySBTTZIIVFJeBui'), (b'\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaachars after 72 are ignored as usual', b'$2a$05$/OK.fbVrR/bpIqNJ5ianF.', b'$2a$05$/OK.fbVrR/bpIqNJ5ianF.swQOIzjOiJ9GHEPuhEkvqrUyvWhEMx6'), (b'\xa3', b'$2a$05$/OK.fbVrR/bpIqNJ5ianF.', b'$2a$05$/OK.fbVrR/bpIqNJ5ianF.Sa7shbm4.OzKpvFnX1pQLmQW96oUlCq'), (b'}>\xb3\xfe\xf1\x8b\xa0\xe6(\xa2Lzq\xc3P\x7f\xcc\xc8b{\xf9\x14\xf6\xf6`\x81G5\xec\x1d\x87\x10\xbf\xa7\xe1}I7 \x96\xdfc\xf2\xbf\xb3Vh\xdfM\x88q\xf7\xff\x1b\x82~z\x13\xdd\xe9\x84\x00\xdd4', b'$2b$10$keO.ZZs22YtygVF6BLfhGO', b'$2b$10$keO.ZZs22YtygVF6BLfhGOI/JjshJYPp8DZsUtym6mJV2Eha2Hdd.'), (b"g7\r\x01\xf3\xd4\xd0\xa9JB^\x18\x007P\xb2N\xc7\x1c\xee\x87&\x83C\x8b\xe8\x18\xc5>\x86\x14/\xd6\xcc\x1cJ\xde\xd7ix\xeb\xdeO\xef\xe1i\xac\xcb\x03\x96v1' \xd6@.m\xa5!\xa0\xef\xc0(", b'$2a$04$tecY.9ylRInW/rAAzXCXPO', b'$2a$04$tecY.9ylRInW/rAAzXCXPOOlyYeCNzmNTzPDNSIFztFMKbvs/s5XG')]
_2y_test_vectors = [(b'\xa3', b'$2y$05$/OK.fbVrR/bpIqNJ5ianF.Sa7shbm4.OzKpvFnX1pQLmQW96oUlCq', b'$2y$05$/OK.fbVrR/bpIqNJ5ianF.Sa7shbm4.OzKpvFnX1pQLmQW96oUlCq'), (b'\xff\xff\xa3', b'$2y$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e', b'$2y$05$/OK.fbVrR/bpIqNJ5ianF.CE5elHaaO4EbggVDjb8P19RukzXSM3e')]

@run_in_pyodide(packages=['bcrypt'])
def test_gensalt_basic(selenium, monkeypatch):
    if False:
        print('Hello World!')
    import os
    import bcrypt
    orig_urandom = os.urandom
    try:
        os.urandom = lambda n: b'0000000000000000'
        assert bcrypt.gensalt() == b'$2b$12$KB.uKB.uKB.uKB.uKB.uK.'
    finally:
        os.urandom = orig_urandom

@pytest.mark.parametrize(('rounds', 'expected'), [(4, b'$2b$04$KB.uKB.uKB.uKB.uKB.uK.'), (5, b'$2b$05$KB.uKB.uKB.uKB.uKB.uK.'), (6, b'$2b$06$KB.uKB.uKB.uKB.uKB.uK.'), (7, b'$2b$07$KB.uKB.uKB.uKB.uKB.uK.'), (8, b'$2b$08$KB.uKB.uKB.uKB.uKB.uK.'), (9, b'$2b$09$KB.uKB.uKB.uKB.uKB.uK.'), (10, b'$2b$10$KB.uKB.uKB.uKB.uKB.uK.'), (11, b'$2b$11$KB.uKB.uKB.uKB.uKB.uK.'), (12, b'$2b$12$KB.uKB.uKB.uKB.uKB.uK.'), (13, b'$2b$13$KB.uKB.uKB.uKB.uKB.uK.'), (14, b'$2b$14$KB.uKB.uKB.uKB.uKB.uK.'), (15, b'$2b$15$KB.uKB.uKB.uKB.uKB.uK.'), (16, b'$2b$16$KB.uKB.uKB.uKB.uKB.uK.'), (17, b'$2b$17$KB.uKB.uKB.uKB.uKB.uK.'), (18, b'$2b$18$KB.uKB.uKB.uKB.uKB.uK.'), (19, b'$2b$19$KB.uKB.uKB.uKB.uKB.uK.'), (20, b'$2b$20$KB.uKB.uKB.uKB.uKB.uK.'), (21, b'$2b$21$KB.uKB.uKB.uKB.uKB.uK.'), (22, b'$2b$22$KB.uKB.uKB.uKB.uKB.uK.'), (23, b'$2b$23$KB.uKB.uKB.uKB.uKB.uK.'), (24, b'$2b$24$KB.uKB.uKB.uKB.uKB.uK.')])
@run_in_pyodide(packages=['bcrypt'])
def test_gensalt_rounds_valid(selenium, rounds, expected):
    if False:
        i = 10
        return i + 15
    import os
    import bcrypt
    orig_urandom = os.urandom
    try:
        os.urandom = lambda n: b'0000000000000000'
        assert bcrypt.gensalt(rounds) == expected
    finally:
        os.urandom = orig_urandom

@pytest.mark.parametrize('rounds', list(range(1, 4)))
@run_in_pyodide(packages=['bcrypt'])
def test_gensalt_rounds_invalid(selenium, rounds):
    if False:
        while True:
            i = 10
    import bcrypt
    import pytest
    with pytest.raises(ValueError):
        bcrypt.gensalt(rounds)

@run_in_pyodide(packages=['bcrypt'])
def test_gensalt_bad_prefix(selenium):
    if False:
        print('Hello World!')
    import bcrypt
    import pytest
    with pytest.raises(ValueError):
        bcrypt.gensalt(prefix='bad')

@run_in_pyodide(packages=['bcrypt'])
def test_gensalt_2a_prefix(selenium):
    if False:
        i = 10
        return i + 15
    import os
    import bcrypt
    orig_urandom = os.urandom
    try:
        os.urandom = lambda n: b'0000000000000000'
        assert bcrypt.gensalt(prefix=b'2a') == b'$2a$12$KB.uKB.uKB.uKB.uKB.uK.'
    finally:
        os.urandom = orig_urandom

@pytest.mark.parametrize(('password', 'salt', 'hashed'), _test_vectors)
@run_in_pyodide(packages=['bcrypt'])
def test_hashpw_new(selenium, password, salt, hashed):
    if False:
        while True:
            i = 10
    import bcrypt
    assert bcrypt.hashpw(password, salt) == hashed

@pytest.mark.parametrize(('password', 'salt', 'hashed'), _test_vectors)
@run_in_pyodide(packages=['bcrypt'])
def test_checkpw(selenium, password, salt, hashed):
    if False:
        for i in range(10):
            print('nop')
    import bcrypt
    assert bcrypt.checkpw(password, hashed) is True

@pytest.mark.parametrize(('password', 'salt', 'hashed'), _test_vectors)
@run_in_pyodide(packages=['bcrypt'])
def test_hashpw_existing(selenium, password, salt, hashed):
    if False:
        i = 10
        return i + 15
    import bcrypt
    assert bcrypt.hashpw(password, hashed) == hashed

@pytest.mark.parametrize(('password', 'hashed', 'expected'), _2y_test_vectors)
@run_in_pyodide(packages=['bcrypt'])
def test_hashpw_2y_prefix(selenium, password, hashed, expected):
    if False:
        while True:
            i = 10
    import bcrypt
    assert bcrypt.hashpw(password, hashed) == expected

@pytest.mark.parametrize(('password', 'hashed', 'expected'), _2y_test_vectors)
@run_in_pyodide(packages=['bcrypt'])
def test_checkpw_2y_prefix(selenium, password, hashed, expected):
    if False:
        i = 10
        return i + 15
    import bcrypt
    assert bcrypt.checkpw(password, hashed) is True

@run_in_pyodide(packages=['bcrypt'])
def test_hashpw_invalid(selenium):
    if False:
        i = 10
        return i + 15
    import bcrypt
    import pytest
    with pytest.raises(ValueError):
        bcrypt.hashpw(b'password', b'$2z$04$cVWp4XaNU8a4v1uMRum2SO')

@run_in_pyodide(packages=['bcrypt'])
def test_checkpw_wrong_password(selenium):
    if False:
        print('Hello World!')
    import bcrypt
    assert bcrypt.checkpw(b'badpass', b'$2b$04$2Siw3Nv3Q/gTOIPetAyPr.GNj3aO0lb1E5E9UumYGKjP9BYqlNWJe') is False

@run_in_pyodide(packages=['bcrypt'])
def test_checkpw_bad_salt(selenium):
    if False:
        return 10
    import bcrypt
    import pytest
    with pytest.raises(ValueError):
        bcrypt.checkpw(b'badpass', b'$2b$04$?Siw3Nv3Q/gTOIPetAyPr.GNj3aO0lb1E5E9UumYGKjP9BYqlNWJe')

@run_in_pyodide(packages=['bcrypt'])
def test_checkpw_str_password(selenium):
    if False:
        i = 10
        return i + 15
    import bcrypt
    import pytest
    with pytest.raises(TypeError):
        bcrypt.checkpw('password', b'$2b$04$cVWp4XaNU8a4v1uMRum2SO')

@run_in_pyodide(packages=['bcrypt'])
def test_checkpw_str_salt(selenium):
    if False:
        return 10
    import bcrypt
    import pytest
    with pytest.raises(TypeError):
        bcrypt.checkpw(b'password', '$2b$04$cVWp4XaNU8a4v1uMRum2SO')

@run_in_pyodide(packages=['bcrypt'])
def test_hashpw_str_password(selenium):
    if False:
        return 10
    import bcrypt
    import pytest
    with pytest.raises(TypeError):
        bcrypt.hashpw('password', b'$2b$04$cVWp4XaNU8a4v1uMRum2SO')

@run_in_pyodide(packages=['bcrypt'])
def test_hashpw_str_salt(selenium):
    if False:
        for i in range(10):
            print('nop')
    import bcrypt
    import pytest
    with pytest.raises(TypeError):
        bcrypt.hashpw(b'password', '$2b$04$cVWp4XaNU8a4v1uMRum2SO')

@run_in_pyodide(packages=['bcrypt'])
def test_checkpw_nul_byte(selenium):
    if False:
        for i in range(10):
            print('nop')
    import bcrypt
    import pytest
    bcrypt.checkpw(b'abc\x00def', b'$2b$04$2Siw3Nv3Q/gTOIPetAyPr.GNj3aO0lb1E5E9UumYGKjP9BYqlNWJe')
    with pytest.raises(ValueError):
        bcrypt.checkpw(b'abcdef', b'$2b$04$2S\x00w3Nv3Q/gTOIPetAyPr.GNj3aO0lb1E5E9UumYGKjP9BYqlNWJe')

@run_in_pyodide(packages=['bcrypt'])
def test_hashpw_nul_byte(selenium):
    if False:
        i = 10
        return i + 15
    import bcrypt
    salt = bcrypt.gensalt(4)
    hashed = bcrypt.hashpw(b'abc\x00def', salt)
    assert bcrypt.checkpw(b'abc\x00def', hashed)
    assert not bcrypt.checkpw(b'abc\x00deg', hashed)
    assert not bcrypt.checkpw(b'abc\x00def\x00', hashed)
    assert not bcrypt.checkpw(b'abc\x00def\x00\x00', hashed)

@run_in_pyodide(packages=['bcrypt'])
def test_checkpw_extra_data(selenium):
    if False:
        for i in range(10):
            print('nop')
    import bcrypt
    salt = bcrypt.gensalt(4)
    hashed = bcrypt.hashpw(b'abc', salt)
    assert bcrypt.checkpw(b'abc', hashed)
    assert bcrypt.checkpw(b'abc', hashed + b'extra') is False
    assert bcrypt.checkpw(b'abc', hashed[:-10]) is False

@pytest.mark.parametrize(('rounds', 'password', 'salt', 'expected'), [[4, b'password', b'salt', b"[\xbf\x0c\xc2\x93X\x7f\x1c65U\\'ye\x98\xd4~W\x90q\xbfB~\x9d\x8f\xbe\x84*\xba4\xd9"], [4, b'password', b'\x00', b'\xc1+Vb5\xee\xe0L!%\x98\x97\nW\x9ag'], [4, b'\x00', b'salt', b'`Q\xbe\x18\xc2\xf4\xf8,\xbf\x0e\xfe\xe5G\x1bK\xb9'], [4, b'password\x00', b'salt\x00', b't\x10\xe4L\xf4\xfa\x07\xbf\xaa\xc8\xa9(\xb1r\x7f\xac\x00\x13u\xe7\xbfs\x847\x0fH\xef\xd1!t0P'], [4, b'pass\x00wor', b'sa\x00l', b'\xc2\xbf\xfd\x9d\xb3\x8fei\xef\xefCr\xf4\xde\x83\xc0'], [4, b'pass\x00word', b'sa\x00lt', b'K\xa4\xac9%\xc0\xe8\xd7\xf0\xcd\xb6\xbb\x16\x84\xa5o'], [8, b'password', b'salt', b"\xe16~\xc5\x15\x1a3\xfa\xacL\xc1\xc1D\xcd#\xfa\x15\xd5T\x84\x93\xec\xc9\x9b\x9b]\x9c\r;'\xbe\xc7b'\xeaf\x08\x8b\x84\x9b \xabz\xa4x\x01\x02F\xe7K\xbaQr?\xef\xa9\xf9GMe\x08\x84^\x8d"], [42, b'password', b'salt', b'\x83<\xf0\xdc\xf5m\xb6V\x08\xe8\xf0\xdc\x0c\xe8\x82\xbd'], [8, b'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.', b'salis\x00', b'\x10\x97\x8b\x07%=\xf5\x7fq\xa1b\xeb\x0e\x8a\xd3\n'], [8, b'\r\xb3\xac\x94\xb3\xeeS(OJ"\x89;<$\xae', b':b\xf0\xf0\xdb\xce\xf8#\xcf\xcc\x85HV\xea\x10(', b' D8\x17^\xee|\xe16\xc9\x1bI\xa6y#\xff'], [8, b'\r\xb3\xac\x94\xb3\xeeS(OJ"\x89;<$\xae', b':b\xf0\xf0\xdb\xce\xf8#\xcf\xcc\x85HV\xea\x10(', b" T\xb9\xff\xf3N7!D\x034th(\xe9\xed8\xdeKr\xe0\xa6\x9a\xdc\x17\n\x13\xb5\xe8\xd6F8^\xa4\x03J\xe6\xd2f\x00\xee#2\xc5\xed@\xadU|\x86\xe3@?\xbb0\xe4\xe1\xdc\x1a\xe0k\x99\xa0q6\x8fQ\x8d,BfQ\xc9\xe7\xe47\xfdl\x91[\x1b\xbf\xc3\xa4\xce\xa7\x14\x91I\x0e\xa7\xaf\xb7\xdd\x02\x90\xa6x\xa4\xf4A\x12\x8d\xb1y.\xab'v\xb2\x1e\xb4#\x8e\x07\x15\xad\xd4\x12}\xffD\xe4\xb3\xe4\xccLO\x99p\x08??t\xbdi\x88s\xfd\xf6H\x84Ou\xc9\xbf\x7f\x9e\x0cM\x9e]\x89\xa7x9\x97I)fag\x07a\x1c\xb9\x01\xde1\xa1\x97&\xb6\xe0\x8c:\x80\x01f\x1f-\\\x9d\xcc3\xb4\xaa\x07/\x90\xdd\x0b?T\x8d^\xeb\xa4!\x13\x97\xe2\xfb\x06.Rn\x1dh\xf4jL\xe2V\x18[K\xad\xc2h_\xbex\xe1\xc7e{Y\xf8:\xb9\xab\x80\xcf\x93\x18\xd6\xad\xd1\xf5\x93?\x12\xd6\xf3a\x82\xc8\xe8\x11_h\x03\n\x12D"], [8, b'\xe1\xbd\x88\xce\xb4\xcf\x85\xcf\x83\xcf\x83\xce\xb5\xcf\x8d\xcf\x82', b'\xce\xa4\xce\xb7\xce\xbb\xce\xad\xce\xbc\xce\xb1\xcf\x87\xce\xbf\xcf\x82', b"Cfl\x9b\t\xef3\xed\x8c'\xe8\xe8\xf3\xe2\xd8\xe6"]])
@run_in_pyodide(packages=['bcrypt'])
def test_kdf(selenium, rounds, password, salt, expected):
    if False:
        print('Hello World!')
    import bcrypt
    derived = bcrypt.kdf(password, salt, len(expected), rounds, ignore_few_rounds=True)
    assert derived == expected

@run_in_pyodide(packages=['bcrypt'])
def test_kdf_str_password(selenium):
    if False:
        print('Hello World!')
    import bcrypt
    import pytest
    with pytest.raises(TypeError):
        bcrypt.kdf('password', b'$2b$04$cVWp4XaNU8a4v1uMRum2SO', 10, 10)

@run_in_pyodide(packages=['bcrypt'])
def test_kdf_str_salt(selenium):
    if False:
        i = 10
        return i + 15
    import bcrypt
    import pytest
    with pytest.raises(TypeError):
        bcrypt.kdf(b'password', 'salt', 10, 10)

@run_in_pyodide(packages=['bcrypt'])
def test_kdf_no_warn_rounds(selenium):
    if False:
        for i in range(10):
            print('nop')
    import bcrypt
    bcrypt.kdf(b'password', b'salt', 10, 10, True)

@run_in_pyodide(packages=['bcrypt'])
def test_kdf_warn_rounds(selenium):
    if False:
        print('Hello World!')
    import bcrypt
    import pytest
    with pytest.warns(UserWarning):
        bcrypt.kdf(b'password', b'salt', 10, 10)

@pytest.mark.parametrize(('password', 'salt', 'desired_key_bytes', 'rounds', 'error'), [('pass', b'$2b$04$cVWp4XaNU8a4v1uMRum2SO', 10, 10, TypeError), (b'password', 'salt', 10, 10, TypeError), (b'', b'$2b$04$cVWp4XaNU8a4v1uMRum2SO', 10, 10, ValueError), (b'password', b'', 10, 10, ValueError), (b'password', b'$2b$04$cVWp4XaNU8a4v1uMRum2SO', 0, 10, ValueError), (b'password', b'$2b$04$cVWp4XaNU8a4v1uMRum2SO', -3, 10, ValueError), (b'password', b'$2b$04$cVWp4XaNU8a4v1uMRum2SO', 513, 10, ValueError), (b'password', b'$2b$04$cVWp4XaNU8a4v1uMRum2SO', 20, 0, ValueError)])
@run_in_pyodide(packages=['bcrypt'])
def test_invalid_params(selenium, password, salt, desired_key_bytes, rounds, error):
    if False:
        print('Hello World!')
    import bcrypt
    import pytest
    with pytest.raises(error):
        bcrypt.kdf(password, salt, desired_key_bytes, rounds)

@run_in_pyodide(packages=['bcrypt'])
def test_2a_wraparound_bug(selenium):
    if False:
        print('Hello World!')
    import bcrypt
    assert bcrypt.hashpw((b'0123456789' * 26)[:255], b'$2a$04$R1lJ2gkNaoPGdafE.H.16.') == b'$2a$04$R1lJ2gkNaoPGdafE.H.16.1MKHPvmKwryeulRe225LKProWYwt9Oi'