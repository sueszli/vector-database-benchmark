"""
tests.pytests.unit.test_crypt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for salt's crypt module
"""
import os
import uuid
import pytest
import salt.crypt
import salt.master
import salt.utils.files
from tests.support.helpers import dedent
from tests.support.mock import MagicMock, MockCall, mock_open, patch
try:
    import M2Crypto
    HAS_M2 = True
except ImportError:
    HAS_M2 = False
try:
    from Cryptodome.PublicKey import RSA
    HAS_PYCRYPTO_RSA = True
except ImportError:
    HAS_PYCRYPTO_RSA = False
if not HAS_PYCRYPTO_RSA:
    try:
        from Crypto.PublicKey import RSA
        HAS_PYCRYPTO_RSA = True
    except ImportError:
        HAS_PYCRYPTO_RSA = False
PRIV_KEY = '\n-----BEGIN RSA PRIVATE KEY-----\nMIIEogIBAAKCAQEAoAsMPt+4kuIG6vKyw9r3+OuZrVBee/2vDdVetW+Js5dTlgrJ\naghWWn3doGmKlEjqh7E4UTa+t2Jd6w8RSLnyHNJ/HpVhMG0M07MF6FMfILtDrrt8\nZX7eDVt8sx5gCEpYI+XG8Y07Ga9i3Hiczt+fu6HYwu96HggmG2pqkOrn3iGfqBvV\nYVFJzSZYe7e4c1PeEs0xYcrA4k+apyGsMtpef8vRUrNicRLc7dAcvfhtgt2DXEZ2\nd72t/CR4ygtUvPXzisaTPW0G7OWAheCloqvTIIPQIjR8htFxGTz02STVXfnhnJ0Z\nk8KhqKF2v1SQvIYxsZU7jaDgl5i3zpeh58cYOwIDAQABAoIBABZUJEO7Y91+UnfC\nH6XKrZEZkcnH7j6/UIaOD9YhdyVKxhsnax1zh1S9vceNIgv5NltzIsfV6vrb6v2K\nDx/F7Z0O0zR5o+MlO8ZncjoNKskex10gBEWG00Uqz/WPlddiQ/TSMJTv3uCBAzp+\nS2Zjdb4wYPUlgzSgb2ygxrhsRahMcSMG9PoX6klxMXFKMD1JxiY8QfAHahPzQXy9\nF7COZ0fCVo6BE+MqNuQ8tZeIxu8mOULQCCkLFwXmkz1FpfK/kNRmhIyhxwvCS+z4\nJuErW3uXfE64RLERiLp1bSxlDdpvRO2R41HAoNELTsKXJOEt4JANRHm/CeyA5wsh\nNpscufUCgYEAxhgPfcMDy2v3nL6KtkgYjdcOyRvsAF50QRbEa8ldO+87IoMDD/Oe\nosFERJ5hhyyEO78QnaLVegnykiw5DWEF02RKMhD/4XU+1UYVhY0wJjKQIBadsufB\n2dnaKjvwzUhPh5BrBqNHl/FXwNCRDiYqXa79eWCPC9OFbZcUWWq70s8CgYEAztOI\n61zRfmXJ7f70GgYbHg+GA7IrsAcsGRITsFR82Ho0lqdFFCxz7oK8QfL6bwMCGKyk\nnzk+twh6hhj5UNp18KN8wktlo02zTgzgemHwaLa2cd6xKgmAyuPiTgcgnzt5LVNG\nFOjIWkLwSlpkDTl7ZzY2QSy7t+mq5d750fpIrtUCgYBWXZUbcpPL88WgDB7z/Bjg\ndlvW6JqLSqMK4b8/cyp4AARbNp12LfQC55o5BIhm48y/M70tzRmfvIiKnEc/gwaE\nNJx4mZrGFFURrR2i/Xx5mt/lbZbRsmN89JM+iKWjCpzJ8PgIi9Wh9DIbOZOUhKVB\n9RJEAgo70LvCnPTdS0CaVwKBgDJW3BllAvw/rBFIH4OB/vGnF5gosmdqp3oGo1Ik\njipmPAx6895AH4tquIVYrUl9svHsezjhxvjnkGK5C115foEuWXw0u60uiTiy+6Pt\n2IS0C93VNMulenpnUrppE7CN2iWFAiaura0CY9fE/lsVpYpucHAWgi32Kok+ZxGL\nWEttAoGAN9Ehsz4LeQxEj3x8wVeEMHF6OsznpwYsI2oVh6VxpS4AjgKYqeLVcnNi\nTlZFsuQcqgod8OgzA91tdB+Rp86NygmWD5WzeKXpCOg9uA+y/YL+0sgZZHsuvbK6\nPllUgXdYxqClk/hdBFB7v9AQoaj7K9Ga22v32msftYDQRJ94xOI=\n-----END RSA PRIVATE KEY-----\n'
PUB_KEY = '\n-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAoAsMPt+4kuIG6vKyw9r3\n+OuZrVBee/2vDdVetW+Js5dTlgrJaghWWn3doGmKlEjqh7E4UTa+t2Jd6w8RSLny\nHNJ/HpVhMG0M07MF6FMfILtDrrt8ZX7eDVt8sx5gCEpYI+XG8Y07Ga9i3Hiczt+f\nu6HYwu96HggmG2pqkOrn3iGfqBvVYVFJzSZYe7e4c1PeEs0xYcrA4k+apyGsMtpe\nf8vRUrNicRLc7dAcvfhtgt2DXEZ2d72t/CR4ygtUvPXzisaTPW0G7OWAheCloqvT\nIIPQIjR8htFxGTz02STVXfnhnJ0Zk8KhqKF2v1SQvIYxsZU7jaDgl5i3zpeh58cY\nOwIDAQAB\n-----END PUBLIC KEY-----\n'
PRIV_KEY2 = '\n-----BEGIN RSA PRIVATE KEY-----\nMIIEogIBAAKCAQEAp+8cTxguO6Vg+YO92VfHgNld3Zy8aM3JbZvpJcjTnis+YFJ7\nZlkcc647yPRRwY9nYBNywahnt5kIeuT1rTvTsMBZWvmUoEVUj1Xg8XXQkBvb9Ozy\nGqy/G/p8KDDpzMP/U+XCnUeHiXTZrgnqgBIc2cKeCVvWFqDi0GRFGzyaXLaX3PPm\nM7DJ0MIPL1qgmcDq6+7Ze0gJ9SrDYFAeLmbuT1OqDfufXWQl/82JXeiwU2cOpqWq\n7n5fvPOWim7l1tzQ+dSiMRRm0xa6uNexCJww3oJSwvMbAmgzvOhqqhlqv+K7u0u7\nFrFFojESsL36Gq4GBrISnvu2tk7u4GGNTYYQbQIDAQABAoIBAADrqWDQnd5DVZEA\nlR+WINiWuHJAy/KaIC7K4kAMBgbxrz2ZbiY9Ok/zBk5fcnxIZDVtXd1sZicmPlro\nGuWodIxdPZAnWpZ3UtOXUayZK/vCP1YsH1agmEqXuKsCu6Fc+K8VzReOHxLUkmXn\nFYM+tixGahXcjEOi/aNNTWitEB6OemRM1UeLJFzRcfyXiqzHpHCIZwBpTUAsmzcG\nQiVDkMTKubwo/m+PVXburX2CGibUydctgbrYIc7EJvyx/cpRiPZXo1PhHQWdu4Y1\nSOaC66WLsP/wqvtHo58JQ6EN/gjSsbAgGGVkZ1xMo66nR+pLpR27coS7o03xCks6\nDY/0mukCgYEAuLIGgBnqoh7YsOBLd/Bc1UTfDMxJhNseo+hZemtkSXz2Jn51322F\nZw/FVN4ArXgluH+XsOhvG/MFFpojwZSrb0Qq5b1MRdo9qycq8lGqNtlN1WHqosDQ\nzW29kpL0tlRrSDpww3wRESsN9rH5XIrJ1b3ZXuO7asR+KBVQMy/+NcUCgYEA6MSC\nc+fywltKPgmPl5j0DPoDe5SXE/6JQy7w/vVGrGfWGf/zEJmhzS2R+CcfTTEqaT0T\nYw8+XbFgKAqsxwtE9MUXLTVLI3sSUyE4g7blCYscOqhZ8ItCUKDXWkSpt++rG0Um\n1+cEJP/0oCazG6MWqvBC4NpQ1nzh46QpjWqMwokCgYAKDLXJ1p8rvx3vUeUJW6zR\ndfPlEGCXuAyMwqHLxXgpf4EtSwhC5gSyPOtx2LqUtcrnpRmt6JfTH4ARYMW9TMef\nQEhNQ+WYj213mKP/l235mg1gJPnNbUxvQR9lkFV8bk+AGJ32JRQQqRUTbU+yN2MQ\nHEptnVqfTp3GtJIultfwOQKBgG+RyYmu8wBP650izg33BXu21raEeYne5oIqXN+I\nR5DZ0JjzwtkBGroTDrVoYyuH1nFNEh7YLqeQHqvyufBKKYo9cid8NQDTu+vWr5UK\ntGvHnwdKrJmM1oN5JOAiq0r7+QMAOWchVy449VNSWWV03aeftB685iR5BXkstbIQ\nEVopAoGAfcGBTAhmceK/4Q83H/FXBWy0PAa1kZGg/q8+Z0KY76AqyxOVl0/CU/rB\n3tO3sKhaMTHPME/MiQjQQGoaK1JgPY6JHYvly2KomrJ8QTugqNGyMzdVJkXAK2AM\nGAwC8ivAkHf8CHrHa1W7l8t2IqBjW1aRt7mOW92nfG88Hck0Mbo=\n-----END RSA PRIVATE KEY-----\n'
PUB_KEY2 = '\n-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAp+8cTxguO6Vg+YO92VfH\ngNld3Zy8aM3JbZvpJcjTnis+YFJ7Zlkcc647yPRRwY9nYBNywahnt5kIeuT1rTvT\nsMBZWvmUoEVUj1Xg8XXQkBvb9OzyGqy/G/p8KDDpzMP/U+XCnUeHiXTZrgnqgBIc\n2cKeCVvWFqDi0GRFGzyaXLaX3PPmM7DJ0MIPL1qgmcDq6+7Ze0gJ9SrDYFAeLmbu\nT1OqDfufXWQl/82JXeiwU2cOpqWq7n5fvPOWim7l1tzQ+dSiMRRm0xa6uNexCJww\n3oJSwvMbAmgzvOhqqhlqv+K7u0u7FrFFojESsL36Gq4GBrISnvu2tk7u4GGNTYYQ\nbQIDAQAB\n-----END PUBLIC KEY-----\n'
PRIVKEY_DATA = '-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA75GR6ZTv5JOv90Vq8tKhKC7YQnhDIo2hM0HVziTEk5R4UQBW\na0CKytFMbTONY2msEDwX9iA0x7F5Lgj0X8eD4ZMsYqLzqjWMekLC8bjhxc+EuPo9\nDygu3mJ2VgRC7XhlFpmdo5NN8J2E7B/CNB3R4hOcMMZNZdi0xLtFoTfwU61UPfFX\n14mV2laqLbvDEfQLJhUTDeFFV8EN5Z4H1ttLP3sMXJvc3EvM0JiDVj4l1TWFUHHz\neFgCA1Im0lv8i7PFrgW7nyMfK9uDSsUmIp7k6ai4tVzwkTmV5PsriP1ju88Lo3MB\n4/sUmDv/JmlZ9YyzTO3Po8Uz3Aeq9HJWyBWHAQIDAQABAoIBAGOzBzBYZUWRGOgl\nIY8QjTT12dY/ymC05GM6gMobjxuD7FZ5d32HDLu/QrknfS3kKlFPUQGDAbQhbbb0\nzw6VL5NO9mfOPO2W/3FaG1sRgBQcerWonoSSSn8OJwVBHMFLG3a+U1Zh1UvPoiPK\nS734swIM+zFpNYivGPvOm/muF/waFf8tF/47t1cwt/JGXYQnkG/P7z0vp47Irpsb\nYjw7vPe4BnbY6SppSxscW3KoV7GtJLFKIxAXbxsuJMF/rYe3O3w2VKJ1Sug1VDJl\n/GytwAkSUer84WwP2b07Wn4c5pCnmLslMgXCLkENgi1NnJMhYVOnckxGDZk54hqP\n9RbLnkkCgYEA/yKuWEvgdzYRYkqpzB0l9ka7Y00CV4Dha9Of6GjQi9i4VCJ/UFVr\nUlhTo5y0ZzpcDAPcoZf5CFZsD90a/BpQ3YTtdln2MMCL/Kr3QFmetkmDrt+3wYnX\nsKESfsa2nZdOATRpl1antpwyD4RzsAeOPwBiACj4fkq5iZJBSI0bxrMCgYEA8GFi\nqAjgKh81/Uai6KWTOW2kX02LEMVRrnZLQ9VPPLGid4KZDDk1/dEfxjjkcyOxX1Ux\nKlu4W8ZEdZyzPcJrfk7PdopfGOfrhWzkREK9C40H7ou/1jUecq/STPfSOmxh3Y+D\nifMNO6z4sQAHx8VaHaxVsJ7SGR/spr0pkZL+NXsCgYEA84rIgBKWB1W+TGRXJzdf\nyHIGaCjXpm2pQMN3LmP3RrcuZWm0vBt94dHcrR5l+u/zc6iwEDTAjJvqdU4rdyEr\ntfkwr7v6TNlQB3WvpWanIPyVzfVSNFX/ZWSsAgZvxYjr9ixw6vzWBXOeOb/Gqu7b\ncvpLkjmJ0wxDhbXtyXKhZA8CgYBZyvcQb+hUs732M4mtQBSD0kohc5TsGdlOQ1AQ\nMcFcmbpnzDghkclyW8jzwdLMk9uxEeDAwuxWE/UEvhlSi6qdzxC+Zifp5NBc0fVe\n7lMx2mfJGxj5CnSqQLVdHQHB4zSXkAGB6XHbBd0MOUeuvzDPfs2voVQ4IG3FR0oc\n3/znuwKBgQChZGH3McQcxmLA28aUwOVbWssfXKdDCsiJO+PEXXlL0maO3SbnFn+Q\nTyf8oHI5cdP7AbwDSx9bUfRPjg9dKKmATBFr2bn216pjGxK0OjYOCntFTVr0psRB\nCrKg52Qrq71/2l4V2NLQZU40Dr1bN9V+Ftd9L0pvpCAEAWpIbLXGDw==\n-----END RSA PRIVATE KEY-----'
PUBKEY_DATA = '-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA75GR6ZTv5JOv90Vq8tKh\nKC7YQnhDIo2hM0HVziTEk5R4UQBWa0CKytFMbTONY2msEDwX9iA0x7F5Lgj0X8eD\n4ZMsYqLzqjWMekLC8bjhxc+EuPo9Dygu3mJ2VgRC7XhlFpmdo5NN8J2E7B/CNB3R\n4hOcMMZNZdi0xLtFoTfwU61UPfFX14mV2laqLbvDEfQLJhUTDeFFV8EN5Z4H1ttL\nP3sMXJvc3EvM0JiDVj4l1TWFUHHzeFgCA1Im0lv8i7PFrgW7nyMfK9uDSsUmIp7k\n6ai4tVzwkTmV5PsriP1ju88Lo3MB4/sUmDv/JmlZ9YyzTO3Po8Uz3Aeq9HJWyBWH\nAQIDAQAB\n-----END PUBLIC KEY-----'
MSG = b"It's me, Mario"
SIG = b'\x07\xf3\xb1\xe7\xdb\x06\xf4_\xe2\xdc\xcb!F\xfb\xbex{W\x1d\xe4E\xd3\r\xc5\x90\xca(\x05\x1d\x99\x8b\x1aug\x9f\x95>\x94\x7f\xe3+\x12\xfa\x9c\xd4\xb8\x02]\x0e\xa5\xa3LL\xc3\xa2\x8f+\x83Z\x1b\x17\xbfT\xd3\xc7\xfd\x0b\xf4\xd7J\xfe^\x86q"I\xa3x\xbc\xd3$\xe9M<\xe1\x07\xad\xf2_\x9f\xfa\xf7g(~\xd8\xf5\xe7\xda-\xa3Ko\xfc.\x99\xcf\x9b\xb9\xc1U\x97\x82\'\xcb\xc6\x08\xaa\xa0\xe4\xd0\xc1+\xfc\x86\r\xe4y\xb1#\xd3\x1dS\x96D28\xc4\xd5\r\xd4\x98\x1a44"\xd7\xc2\xb4]\xa7\x0f\xa7Db\x85G\x8c\xd6\x94!\x8af1O\xf6g\xd7\x03\xfd\xb3\xbc\xce\x9f\xe7\x015\xb8\x1d]AHK\xa0\x14m\xda=O\xa7\xde\xf2\xff\x9b\x8e\x83\xc8j\x11\x1a\x98\x85\xde\xc5\x91\x07\x84!\x12^4\xcb\xa8\x98\x8a\x8a&#\xb9(#?\x80\x15\x9eW\xb5\x12\xd1\x95S\xf2<G\xeb\xf1\x14H\xb2\xc4>\xc3A\xed\x86x~\xcfU\xd5Q\xfe~\x10\xd2\x9b'
TEST_KEY = '-----BEGIN RSA PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzLtFhsvfbFDFaUgulSEX\nGl12XriL1DT78Ef2/u8HHaSMmPie37BLWas/zaHwI6066bIyYQJ/nUCahTaoHM7L\nGlWc0wOU6zyfpihCRQHil05Y6F+olFBoZuYbFPvtp7/hJx/D7I/0n2o/c7M5i3Y2\n3sBxAYNooIQHXHUmPQW6C9iu95ylZDW8JQzYy/EI4vCC8yQMdTK8jK1FQV0Sbwny\nqcMxSyAWDoFbnhh2P2TnO8HOWuUOaXR8ZHOJzVcDl+a6ew+medW090x3K5O1f80D\n+WjgnG6b2HG7VQpOCfM2GALD/FrxicPilvZ38X1aLhJuwjmVE4LAAv8DVNJXohaO\nWQIDAQAB\n-----END RSA PUBLIC KEY-----\n'
SIGNATURE = b'w\xac\xfe18o\xeb\xfb\x14+\x9e\xd1\xb7\x7fe}\xec\xd6\xe1P\x9e\xab\xb5\x07\xe0\xc1\xfd\xda#\x04Z\x8d\x7f\x0b\x1f}:~\xb2s\x860u\x02N\xd4q"\xb7\x86*\x8f\x1f\xd0\x9d\x11\x92\xc5~\xa68\xac>\x12H\xc2%y,\xe6\xceU\x1e\xa3?\x0c,\xf0u\xbb\xd0[g_\xdd\x8b\xb0\x95:Y\x18\xa5*\x99\xfd\xf3K\x92\x92 ({\xd1\xff\xd9F\xc8\xd6K\x86e\xf9\xa8\xad\xb0z\xe3\x9dD\xf5k\x8b_<\xe7\xe7\xec\xf3"\'\xd5\xd2M\xb4\xce\x1a\xe3$\x9c\x81\xad\xf9\x11\xf6\xf5>)\xc7\xdd\x03&\xf7\x86@ks\xa6\x05\xc2\xd0\xbd\x1a7\xfc\xde\xe6\xb0\xad!\x12#\xc86Y\xea\xc5\xe3\xe2\xb3\xc9\xaf\xfa\x0c\xf2?\xbf\x93w\x18\x9e\x0b\xa2a\x10:M\x05\x89\xe2W.Q\xe8;yGT\xb1\xf2\xc6A\xd2\xc4\xbeN\xb3\xcfS\xaf\x03f\xe2\xb4)\xe7\xf6\xdbs\xd0Z}8\xa4\xd2\x1fW*\xe6\x1c"\x8b\xd0\x18w\xb9\x7f\x9e\x96\xa3\xd9v\xf7\x833\x8e\x01'

def test_get_rsa_pub_key_bad_key(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    get_rsa_pub_key raises InvalidKeyError when encoutering a bad key\n    '
    key_path = str(tmp_path / 'key')
    with salt.utils.files.fopen(key_path, 'w') as fp:
        fp.write('')
    with pytest.raises(salt.crypt.InvalidKeyError):
        salt.crypt.get_rsa_pub_key(key_path)

def test_cryptical_dumps_no_nonce():
    if False:
        for i in range(10):
            print('nop')
    master_crypt = salt.crypt.Crypticle({}, salt.crypt.Crypticle.generate_key_string())
    data = {'foo': 'bar'}
    ret = master_crypt.dumps(data)
    assert isinstance(ret, bytes)
    une = master_crypt.decrypt(ret)
    une.startswith(master_crypt.PICKLE_PAD)
    assert salt.payload.loads(une[len(master_crypt.PICKLE_PAD):]) == data
    assert master_crypt.loads(ret) == data

def test_cryptical_dumps_valid_nonce():
    if False:
        return 10
    nonce = uuid.uuid4().hex
    master_crypt = salt.crypt.Crypticle({}, salt.crypt.Crypticle.generate_key_string())
    data = {'foo': 'bar'}
    ret = master_crypt.dumps(data, nonce=nonce)
    assert isinstance(ret, bytes)
    une = master_crypt.decrypt(ret)
    une.startswith(master_crypt.PICKLE_PAD)
    nonce_and_data = une[len(master_crypt.PICKLE_PAD):]
    assert nonce_and_data.startswith(nonce.encode())
    assert salt.payload.loads(nonce_and_data[len(nonce):]) == data
    assert master_crypt.loads(ret, nonce=nonce) == data

def test_cryptical_dumps_invalid_nonce():
    if False:
        print('Hello World!')
    nonce = uuid.uuid4().hex
    master_crypt = salt.crypt.Crypticle({}, salt.crypt.Crypticle.generate_key_string())
    data = {'foo': 'bar'}
    ret = master_crypt.dumps(data, nonce=nonce)
    assert isinstance(ret, bytes)
    with pytest.raises(salt.crypt.SaltClientError, match='Nonce verification error'):
        assert master_crypt.loads(ret, nonce='abcde')

def test_verify_signature(tmp_path):
    if False:
        i = 10
        return i + 15
    tmp_path.joinpath('foo.pem').write_text(PRIV_KEY.strip())
    tmp_path.joinpath('foo.pub').write_text(PUB_KEY.strip())
    tmp_path.joinpath('bar.pem').write_text(PRIV_KEY2.strip())
    tmp_path.joinpath('bar.pub').write_text(PUB_KEY2.strip())
    msg = b'foo bar'
    sig = salt.crypt.sign_message(str(tmp_path.joinpath('foo.pem')), msg)
    assert salt.crypt.verify_signature(str(tmp_path.joinpath('foo.pub')), msg, sig)

def test_verify_signature_bad_sig(tmp_path):
    if False:
        while True:
            i = 10
    tmp_path.joinpath('foo.pem').write_text(PRIV_KEY.strip())
    tmp_path.joinpath('foo.pub').write_text(PUB_KEY.strip())
    tmp_path.joinpath('bar.pem').write_text(PRIV_KEY2.strip())
    tmp_path.joinpath('bar.pub').write_text(PUB_KEY2.strip())
    msg = b'foo bar'
    sig = salt.crypt.sign_message(str(tmp_path.joinpath('foo.pem')), msg)
    assert not salt.crypt.verify_signature(str(tmp_path.joinpath('bar.pub')), msg, sig)

def test_read_or_generate_key_string(tmp_path):
    if False:
        while True:
            i = 10
    keyfile = tmp_path / '.aes'
    assert not keyfile.exists()
    first_key = salt.crypt.Crypticle.read_or_generate_key(keyfile)
    assert keyfile.exists()
    second_key = salt.crypt.Crypticle.read_or_generate_key(keyfile)
    assert first_key == second_key
    third_key = salt.crypt.Crypticle.read_or_generate_key(keyfile, remove=True)
    assert second_key != third_key

def test_dropfile_contents(tmp_path, master_opts):
    if False:
        while True:
            i = 10
    salt.crypt.dropfile(str(tmp_path), master_opts['user'], master_id=master_opts['id'])
    with salt.utils.files.fopen(str(tmp_path / '.dfn'), 'r') as fp:
        assert master_opts['id'] == fp.read()

def test_master_keys_without_cluster_id(tmp_path, master_opts):
    if False:
        print('Hello World!')
    master_opts['pki_dir'] = str(tmp_path)
    assert master_opts['cluster_id'] is None
    assert master_opts['cluster_pki_dir'] is None
    mkeys = salt.crypt.MasterKeys(master_opts)
    expected_master_pub = str(tmp_path / 'master.pub')
    expected_master_rsa = str(tmp_path / 'master.pem')
    assert expected_master_pub == mkeys.master_pub_path
    assert expected_master_rsa == mkeys.master_rsa_path
    assert mkeys.cluster_pub_path is None
    assert mkeys.cluster_rsa_path is None
    assert mkeys.pub_path == expected_master_pub
    assert mkeys.rsa_path == expected_master_rsa
    assert mkeys.key == mkeys.master_key

def test_master_keys_with_cluster_id(tmp_path, master_opts):
    if False:
        print('Hello World!')
    master_pki_path = tmp_path / 'master_pki'
    cluster_pki_path = tmp_path / 'cluster_pki'
    master_pki_path.mkdir()
    cluster_pki_path.mkdir()
    (cluster_pki_path / 'peers').mkdir()
    master_opts['pki_dir'] = str(master_pki_path)
    master_opts['cluster_id'] = 'cluster1'
    master_opts['cluster_pki_dir'] = str(cluster_pki_path)
    mkeys = salt.crypt.MasterKeys(master_opts)
    expected_master_pub = str(master_pki_path / 'master.pub')
    expected_master_rsa = str(master_pki_path / 'master.pem')
    expected_cluster_pub = str(cluster_pki_path / 'cluster.pub')
    expected_cluster_rsa = str(cluster_pki_path / 'cluster.pem')
    assert expected_master_pub == mkeys.master_pub_path
    assert expected_master_rsa == mkeys.master_rsa_path
    assert expected_cluster_pub == mkeys.cluster_pub_path
    assert expected_cluster_rsa == mkeys.cluster_rsa_path
    assert mkeys.pub_path == expected_cluster_pub
    assert mkeys.rsa_path == expected_cluster_rsa
    assert mkeys.key == mkeys.cluster_key

@pytest.mark.skipif(not HAS_PYCRYPTO_RSA, reason='pycrypto >= 2.6 is not available')
@pytest.mark.skipif(HAS_M2, reason='m2crypto is used by salt.crypt if installed')
def test_pycrypto_gen_keys():
    if False:
        for i in range(10):
            print('nop')
    open_priv_wb = MockCall(f'/keydir{os.sep}keyname.pem', 'wb+')
    open_pub_wb = MockCall(f'/keydir{os.sep}keyname.pub', 'wb+')
    with patch.multiple(os, umask=MagicMock(), chmod=MagicMock(), access=MagicMock(return_value=True)):
        with patch('salt.utils.files.fopen', mock_open()) as m_open, patch('os.path.isfile', return_value=True):
            result = salt.crypt.gen_keys('/keydir', 'keyname', 2048)
            assert result == f'/keydir{os.sep}keyname.pem', result
            assert open_priv_wb not in m_open.calls
            assert open_pub_wb not in m_open.calls
        with patch('salt.utils.files.fopen', mock_open()) as m_open, patch('os.path.isfile', return_value=False):
            salt.crypt.gen_keys('/keydir', 'keyname', 2048)
            assert open_priv_wb in m_open.calls
            assert open_pub_wb in m_open.calls

@patch('os.umask', MagicMock())
@patch('os.chmod', MagicMock())
@patch('os.chown', MagicMock(), create=True)
@patch('os.access', MagicMock(return_value=True))
@pytest.mark.slow_test
@pytest.mark.skipif(not HAS_PYCRYPTO_RSA, reason='pycrypto >= 2.6 is not available')
@pytest.mark.skipif(HAS_M2, reason='m2crypto is used by salt.crypt if installed')
def test_pycrypto_gen_keys_with_passphrase():
    if False:
        return 10
    key_path = os.path.join(os.sep, 'keydir')
    open_priv_wb = MockCall(os.path.join(key_path, 'keyname.pem'), 'wb+')
    open_pub_wb = MockCall(os.path.join(key_path, 'keyname.pub'), 'wb+')
    with patch('salt.utils.files.fopen', mock_open()) as m_open, patch('os.path.isfile', return_value=True):
        assert salt.crypt.gen_keys(key_path, 'keyname', 2048, passphrase='password') == os.path.join(key_path, 'keyname.pem')
        result = salt.crypt.gen_keys(key_path, 'keyname', 2048, passphrase='password')
        assert result == os.path.join(key_path, 'keyname.pem'), result
        assert open_priv_wb not in m_open.calls
        assert open_pub_wb not in m_open.calls
    with patch('salt.utils.files.fopen', mock_open()) as m_open, patch('os.path.isfile', return_value=False):
        salt.crypt.gen_keys(key_path, 'keyname', 2048)
        assert open_priv_wb in m_open.calls
        assert open_pub_wb in m_open.calls

@pytest.mark.skipif(not HAS_PYCRYPTO_RSA, reason='pycrypto >= 2.6 is not available')
@pytest.mark.skipif(HAS_M2, reason='m2crypto is used by salt.crypt if installed')
def test_pycrypto_sign_message():
    if False:
        return 10
    key = RSA.importKey(PRIVKEY_DATA)
    with patch('salt.crypt.get_rsa_key', return_value=key):
        assert SIG == salt.crypt.sign_message('/keydir/keyname.pem', MSG)

@pytest.mark.skipif(not HAS_PYCRYPTO_RSA, reason='pycrypto >= 2.6 is not available')
@pytest.mark.skipif(HAS_M2, reason='m2crypto is used by salt.crypt if installed')
def test_pycrypto_sign_message_with_passphrase():
    if False:
        for i in range(10):
            print('nop')
    key = RSA.importKey(PRIVKEY_DATA)
    with patch('salt.crypt.get_rsa_key', return_value=key):
        assert SIG == salt.crypt.sign_message('/keydir/keyname.pem', MSG, passphrase='password')

@pytest.mark.skipif(not HAS_PYCRYPTO_RSA, reason='pycrypto >= 2.6 is not available')
@pytest.mark.skipif(HAS_M2, reason='m2crypto is used by salt.crypt if installed')
def test_pycrypto_verify_signature():
    if False:
        print('Hello World!')
    with patch('salt.utils.files.fopen', mock_open(read_data=PUBKEY_DATA)):
        assert salt.crypt.verify_signature('/keydir/keyname.pub', MSG, SIG)

@patch('os.umask', MagicMock())
@patch('os.chmod', MagicMock())
@patch('os.access', MagicMock(return_value=True))
@pytest.mark.skipif(not HAS_M2, reason='m2crypto is not available')
@pytest.mark.slow_test
def test_m2_gen_keys():
    if False:
        return 10
    with patch('M2Crypto.RSA.RSA.save_pem', MagicMock()) as save_pem:
        with patch('M2Crypto.RSA.RSA.save_pub_key', MagicMock()) as save_pub:
            with patch('os.path.isfile', return_value=True):
                assert salt.crypt.gen_keys('/keydir', 'keyname', 2048) == f'/keydir{os.sep}keyname.pem'
                save_pem.assert_not_called()
                save_pub.assert_not_called()
            with patch('os.path.isfile', return_value=False):
                assert salt.crypt.gen_keys('/keydir', 'keyname', 2048) == f'/keydir{os.sep}keyname.pem'
                save_pem.assert_called_once_with(f'/keydir{os.sep}keyname.pem', cipher=None)
                save_pub.assert_called_once_with(f'/keydir{os.sep}keyname.pub')

@patch('os.umask', MagicMock())
@patch('os.chmod', MagicMock())
@patch('os.chown', MagicMock())
@patch('os.access', MagicMock(return_value=True))
@pytest.mark.skipif(not HAS_M2, reason='m2crypto is not available')
@pytest.mark.slow_test
def test_gen_keys_with_passphrase():
    if False:
        return 10
    with patch('M2Crypto.RSA.RSA.save_pem', MagicMock()) as save_pem:
        with patch('M2Crypto.RSA.RSA.save_pub_key', MagicMock()) as save_pub:
            with patch('os.path.isfile', return_value=True):
                assert salt.crypt.gen_keys('/keydir', 'keyname', 2048, passphrase='password') == f'/keydir{os.sep}keyname.pem'
                save_pem.assert_not_called()
                save_pub.assert_not_called()
            with patch('os.path.isfile', return_value=False):
                assert salt.crypt.gen_keys('/keydir', 'keyname', 2048, passphrase='password') == f'/keydir{os.sep}keyname.pem'
                callback = save_pem.call_args[1]['callback']
                save_pem.assert_called_once_with(f'/keydir{os.sep}keyname.pem', cipher='des_ede3_cbc', callback=callback)
                assert callback(None) == b'password'
                save_pub.assert_called_once_with(f'/keydir{os.sep}keyname.pub')

@pytest.mark.skipif(not HAS_M2, reason='m2crypto is not available')
def test_m2_sign_message_with_passphrase():
    if False:
        print('Hello World!')
    key = M2Crypto.RSA.load_key_string(salt.utils.stringutils.to_bytes(PRIVKEY_DATA))
    with patch('salt.crypt.get_rsa_key', return_value=key):
        assert SIG == salt.crypt.sign_message('/keydir/keyname.pem', MSG, passphrase='password')

@pytest.mark.skipif(not HAS_M2, reason='m2crypto is not available')
def test_m2_verify_signature():
    if False:
        return 10
    with patch('salt.utils.files.fopen', mock_open(read_data=salt.utils.stringutils.to_bytes(PUBKEY_DATA))):
        assert salt.crypt.verify_signature('/keydir/keyname.pub', MSG, SIG)

@pytest.mark.skipif(not HAS_M2, reason='m2crypto is not available')
def test_m2_encrypt_decrypt_bin():
    if False:
        print('Hello World!')
    priv_key = M2Crypto.RSA.load_key_string(salt.utils.stringutils.to_bytes(PRIVKEY_DATA))
    pub_key = M2Crypto.RSA.load_pub_key_bio(M2Crypto.BIO.MemoryBuffer(salt.utils.stringutils.to_bytes(PUBKEY_DATA)))
    encrypted = salt.crypt.private_encrypt(priv_key, b'salt')
    decrypted = salt.crypt.public_decrypt(pub_key, encrypted)
    assert b'salt' == decrypted

@pytest.fixture
def key_to_test(tmp_path):
    if False:
        while True:
            i = 10
    key_path = tmp_path / 'cryptodom-3.4.6.pub'
    with salt.utils.files.fopen(key_path, 'wb') as fd:
        fd.write(TEST_KEY.encode())
    return key_path

@pytest.mark.skipif(not HAS_M2, reason='Skip when m2crypto is not installed')
def test_m2_bad_key(key_to_test):
    if False:
        i = 10
        return i + 15
    '\n    Load public key with an invalid header using m2crypto and validate it\n    '
    key = salt.crypt.get_rsa_pub_key(key_to_test)
    assert key.check_key() == 1

@pytest.mark.skipif(HAS_M2, reason='Skip when m2crypto is installed')
def test_pycrypto_bad_key(key_to_test):
    if False:
        print('Hello World!')
    '\n    Load public key with an invalid header and validate it without m2crypto\n    '
    key = salt.crypt.get_rsa_pub_key(key_to_test)
    assert key.can_encrypt()

@pytest.mark.skipif(not HAS_M2, reason='Skip when m2crypto is not installed')
def test_m2crypto_verify_bytes_47124():
    if False:
        print('Hello World!')
    message = salt.utils.stringutils.to_unicode('meh')
    with patch('salt.utils.files.fopen', mock_open(read_data=salt.utils.stringutils.to_bytes(PUBKEY_DATA))):
        salt.crypt.verify_signature('/keydir/keyname.pub', message, SIGNATURE)

@pytest.mark.skipif(not HAS_M2, reason='Skip when m2crypto is not installed')
def test_m2crypto_verify_unicode_47124():
    if False:
        print('Hello World!')
    message = salt.utils.stringutils.to_bytes('meh')
    with patch('salt.utils.files.fopen', mock_open(read_data=salt.utils.stringutils.to_bytes(PUBKEY_DATA))):
        salt.crypt.verify_signature('/keydir/keyname.pub', message, SIGNATURE)

@pytest.mark.skipif(not HAS_M2, reason='Skip when m2crypto is not installed')
def test_m2crypto_sign_bytes_47124():
    if False:
        return 10
    message = salt.utils.stringutils.to_unicode('meh')
    key = M2Crypto.RSA.load_key_string(salt.utils.stringutils.to_bytes(PRIVKEY_DATA))
    with patch('salt.crypt.get_rsa_key', return_value=key):
        signature = salt.crypt.sign_message('/keydir/keyname.pem', message, passphrase='password')
    assert SIGNATURE == signature

@pytest.mark.skipif(not HAS_M2, reason='Skip when m2crypto is not installed')
def test_m2crypto_sign_unicode_47124():
    if False:
        for i in range(10):
            print('nop')
    message = salt.utils.stringutils.to_bytes('meh')
    key = M2Crypto.RSA.load_key_string(salt.utils.stringutils.to_bytes(PRIVKEY_DATA))
    with patch('salt.crypt.get_rsa_key', return_value=key):
        signature = salt.crypt.sign_message('/keydir/keyname.pem', message, passphrase='password')
    assert SIGNATURE == signature

def test_pwdata_decrypt():
    if False:
        i = 10
        return i + 15
    key_string = dedent('-----BEGIN RSA PRIVATE KEY-----\n        MIIEpQIBAAKCAQEAzhBRyyHa7b63RLE71uKMKgrpulcAJjaIaN68ltXcCvy4w9pi\n        Kj+4I3Qp6RvUaHOEmymqyjOMjQc6iwpe0scCFqh3nUk5YYaLZ3WAW0htQVlnesgB\n        ZiBg9PBeTQY/LzqtudL6RCng/AX+fbnCsddlIysRxnUoNVMvz0gAmCY2mnTDjcTt\n        pyxuk2T0AHSHNCKCalm75L1bWDFF+UzFemf536tBfBUGRWR6jWTij85vvCntxHS/\n        HdknaTJ50E7XGVzwBJpCyV4Y2VXuW/3KrCNTqXw+jTmEw0vlcshfDg/vb3IxsUSK\n        5KuHalKq/nUIc+F4QCJOl+A10goGdIfYC1/67QIDAQABAoIBAAOP+qoFWtCTZH22\n        hq9PWVb8u0+yY1lFxhPyDdaZueUiu1r/coUCdv996Z+TEJgBr0AzdzVpsLtbbaKr\n        ujnwoNOdc/vvISPTfKN8P4zUcrcXgZd4z7VhR+vUH/0652q8m/ZDdHorMy2IOP8Z\n        cAk9DQ2PmA4TRm+tkX0G5KO8vWLsK921aRMWdsKJyQ0lYxl7M8JWupFsCJFr/U+8\n        dAVtwnUiS7RnhBABZ1cfNTHYhXVAh4d+a9y/gZ00a66OGqPxiXfhjjDUZ6fGvWKN\n        FlhKWEg6YqIx/H4aNXkLI5Rzzhdx/c2ukNm7+X2veRcAW7bcTwk8wxJxciEP5pBi\n        1el9VE0CgYEA/lbzdE2M4yRBvTfYYC6BqZcn+BqtrAUc2h3fEy+p7lwlet0af1id\n        gWpYpOJyLc0AUfR616/m2y3PwEH/nMKDSTuU7o/qKNtlHW0nQcnhDCjTUydS3+J/\n        JM3dhfgVqi03rjqNcgHA2eOEwcu/OBZtiaC0wqKbuRZRtfGffyoO3ssCgYEAz2iw\n        wqu/NkA+MdQIxz/a3Is7gGwoFu6h7O+XU2uN8Y2++jSBw9AzzWj31YCvyjuJPAE+\n        gxHm6yOnNoLVn423NtibHejhabzHNIK6UImH99bSTKabsxfF2BX6v982BimU1jwc\n        bYykzws37oN/poPb5FTpEiAUrsd2bAMn/1S43icCgYEAulHkY0z0aumCpyUkA8HO\n        BvjOtPiGRcAxFLBRXPLL3+vtIQachLHcIJRRf+jLkDXfiCo7W4pm6iWzTbqLkMEG\n        AD3/qowPFAM1Hct6uL01efzmYsIp+g0o60NMhvnolRQu+Bm4yM30AyqjdHzYBjSX\n        5fyuru8EeSCal1j8aOHcpuUCgYEAhGhDH6Pg59NPYSQJjpm3MMA59hwV473n5Yh2\n        xKyO6zwgRT6r8MPDrkhqnwQONT6Yt5PbwnT1Q/t4zhXsJnWkFwFk1U1MSeJYEa+7\n        HZsPECs2CfT6xPRSO0ac00y+AmUdPT8WruDwfbSdukh8f2MCR9vlBsswKPvxH7dM\n        G3aMplUCgYEAmMFgB/6Ox4OsQPPC6g4G+Ezytkc4iVkMEcjiVWzEsYATITjq3weO\n        /XDGBYJoBhYwWPi9oBufFc/2pNtWy1FKKXPuVyXQATdA0mfEPbtsHjMFQNZbeKnm\n        0na/SysSDCK3P+9ijlbjqLjMmPEmhJxGWTJ7khnTTkfre7/w9ZxJxi8=\n        -----END RSA PRIVATE KEY-----')
    pwdata = b'V\x80+b\xca\x06M\xb6\x12\xc6\xe8\xf2\xb5\xbb\xd8m\xc0\x97\x9a\xeb\xb9q\x19\xc3\xcdi\xb84\x90\xaf\x12kT\xe2@u\xd6\xe8T\x89\xa3\xc7\xb2Y\xd1N\x00\xa9\xc0"\xbe\xed\xb1\xc3\xb7^\xbf\xbd\x8b\x13\xd3/L\x1b\xa1`\xe2\xea\x03\x98\x82\xf3uS&|\xe5\xd8J\xce\xfc\x97\x8d\x0b\x949\xc0\xbd^\xef\xc6\xfd\xce\xbb\x1e\xd0"(m\xe1\x95\xfb\xc8/\x07\x93\xb8\xda\x8f\x99\xfe\xdc\xd5\xcb\xdb\xb2\xf11M\xdbD\xcf\x95\x13p\r\xa4\x1c{\xd5\xdb\xc7\xe5\xaf\x95F\x97\xa9\x00p~\xb5\xec\xa4.\xd0\xa4\xb4\xf4f\xcds,Y/\xa1:WF\xb8\xc7\x07\xaa\x0b<\'~\x1b$D9\xd4\x8d\xf0x\xc5\xee\xa8:\xe6\x00\x10\xc5i\x11\xc7]C8\x05l\x8b\x9b\xc3\x83e\xf7y\xadi:0\xb4R\x1a(\x04&yL8\x19s\n\x11\x81\xfd?\xfb2\x80Ll\xa1\xdc\xc9\xb6P\xca\x8d\'\x11\xc1\x07\xa5\xa1\x058\xc7\xce\xbeb\x92\xbf\x0bL\xec\xdf\xc3M\x83\xfb$\xec\xd5\xf9'
    assert '1234', salt.crypt.pwdata_decrypt(key_string, pwdata)