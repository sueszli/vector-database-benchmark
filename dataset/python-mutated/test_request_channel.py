"""
    :codeauthor: Thomas Jackson <jacksontj.89@gmail.com>
"""
import asyncio
import ctypes
import logging
import multiprocessing
import os
import threading
import time
import uuid
import pytest
import tornado.gen
import tornado.ioloop
import salt.channel.client
import salt.channel.server
import salt.crypt
import salt.transport.zeromq
import salt.utils.process
import salt.utils.stringutils
from salt.master import SMaster
from tests.support.mock import MagicMock, patch
try:
    from M2Crypto import RSA
    HAS_M2 = True
except ImportError:
    HAS_M2 = False
    try:
        from Cryptodome.Cipher import PKCS1_OAEP
    except ImportError:
        from Crypto.Cipher import PKCS1_OAEP
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.core_test]
MASTER_PRIV_KEY = '\n-----BEGIN RSA PRIVATE KEY-----\nMIIEogIBAAKCAQEAoAsMPt+4kuIG6vKyw9r3+OuZrVBee/2vDdVetW+Js5dTlgrJ\naghWWn3doGmKlEjqh7E4UTa+t2Jd6w8RSLnyHNJ/HpVhMG0M07MF6FMfILtDrrt8\nZX7eDVt8sx5gCEpYI+XG8Y07Ga9i3Hiczt+fu6HYwu96HggmG2pqkOrn3iGfqBvV\nYVFJzSZYe7e4c1PeEs0xYcrA4k+apyGsMtpef8vRUrNicRLc7dAcvfhtgt2DXEZ2\nd72t/CR4ygtUvPXzisaTPW0G7OWAheCloqvTIIPQIjR8htFxGTz02STVXfnhnJ0Z\nk8KhqKF2v1SQvIYxsZU7jaDgl5i3zpeh58cYOwIDAQABAoIBABZUJEO7Y91+UnfC\nH6XKrZEZkcnH7j6/UIaOD9YhdyVKxhsnax1zh1S9vceNIgv5NltzIsfV6vrb6v2K\nDx/F7Z0O0zR5o+MlO8ZncjoNKskex10gBEWG00Uqz/WPlddiQ/TSMJTv3uCBAzp+\nS2Zjdb4wYPUlgzSgb2ygxrhsRahMcSMG9PoX6klxMXFKMD1JxiY8QfAHahPzQXy9\nF7COZ0fCVo6BE+MqNuQ8tZeIxu8mOULQCCkLFwXmkz1FpfK/kNRmhIyhxwvCS+z4\nJuErW3uXfE64RLERiLp1bSxlDdpvRO2R41HAoNELTsKXJOEt4JANRHm/CeyA5wsh\nNpscufUCgYEAxhgPfcMDy2v3nL6KtkgYjdcOyRvsAF50QRbEa8ldO+87IoMDD/Oe\nosFERJ5hhyyEO78QnaLVegnykiw5DWEF02RKMhD/4XU+1UYVhY0wJjKQIBadsufB\n2dnaKjvwzUhPh5BrBqNHl/FXwNCRDiYqXa79eWCPC9OFbZcUWWq70s8CgYEAztOI\n61zRfmXJ7f70GgYbHg+GA7IrsAcsGRITsFR82Ho0lqdFFCxz7oK8QfL6bwMCGKyk\nnzk+twh6hhj5UNp18KN8wktlo02zTgzgemHwaLa2cd6xKgmAyuPiTgcgnzt5LVNG\nFOjIWkLwSlpkDTl7ZzY2QSy7t+mq5d750fpIrtUCgYBWXZUbcpPL88WgDB7z/Bjg\ndlvW6JqLSqMK4b8/cyp4AARbNp12LfQC55o5BIhm48y/M70tzRmfvIiKnEc/gwaE\nNJx4mZrGFFURrR2i/Xx5mt/lbZbRsmN89JM+iKWjCpzJ8PgIi9Wh9DIbOZOUhKVB\n9RJEAgo70LvCnPTdS0CaVwKBgDJW3BllAvw/rBFIH4OB/vGnF5gosmdqp3oGo1Ik\njipmPAx6895AH4tquIVYrUl9svHsezjhxvjnkGK5C115foEuWXw0u60uiTiy+6Pt\n2IS0C93VNMulenpnUrppE7CN2iWFAiaura0CY9fE/lsVpYpucHAWgi32Kok+ZxGL\nWEttAoGAN9Ehsz4LeQxEj3x8wVeEMHF6OsznpwYsI2oVh6VxpS4AjgKYqeLVcnNi\nTlZFsuQcqgod8OgzA91tdB+Rp86NygmWD5WzeKXpCOg9uA+y/YL+0sgZZHsuvbK6\nPllUgXdYxqClk/hdBFB7v9AQoaj7K9Ga22v32msftYDQRJ94xOI=\n-----END RSA PRIVATE KEY-----\n'
MASTER_PUB_KEY = '\n-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAoAsMPt+4kuIG6vKyw9r3\n+OuZrVBee/2vDdVetW+Js5dTlgrJaghWWn3doGmKlEjqh7E4UTa+t2Jd6w8RSLny\nHNJ/HpVhMG0M07MF6FMfILtDrrt8ZX7eDVt8sx5gCEpYI+XG8Y07Ga9i3Hiczt+f\nu6HYwu96HggmG2pqkOrn3iGfqBvVYVFJzSZYe7e4c1PeEs0xYcrA4k+apyGsMtpe\nf8vRUrNicRLc7dAcvfhtgt2DXEZ2d72t/CR4ygtUvPXzisaTPW0G7OWAheCloqvT\nIIPQIjR8htFxGTz02STVXfnhnJ0Zk8KhqKF2v1SQvIYxsZU7jaDgl5i3zpeh58cY\nOwIDAQAB\n-----END PUBLIC KEY-----\n'
MASTER2_PRIV_KEY = '\n-----BEGIN RSA PRIVATE KEY-----\nMIIEogIBAAKCAQEAp+8cTxguO6Vg+YO92VfHgNld3Zy8aM3JbZvpJcjTnis+YFJ7\nZlkcc647yPRRwY9nYBNywahnt5kIeuT1rTvTsMBZWvmUoEVUj1Xg8XXQkBvb9Ozy\nGqy/G/p8KDDpzMP/U+XCnUeHiXTZrgnqgBIc2cKeCVvWFqDi0GRFGzyaXLaX3PPm\nM7DJ0MIPL1qgmcDq6+7Ze0gJ9SrDYFAeLmbuT1OqDfufXWQl/82JXeiwU2cOpqWq\n7n5fvPOWim7l1tzQ+dSiMRRm0xa6uNexCJww3oJSwvMbAmgzvOhqqhlqv+K7u0u7\nFrFFojESsL36Gq4GBrISnvu2tk7u4GGNTYYQbQIDAQABAoIBAADrqWDQnd5DVZEA\nlR+WINiWuHJAy/KaIC7K4kAMBgbxrz2ZbiY9Ok/zBk5fcnxIZDVtXd1sZicmPlro\nGuWodIxdPZAnWpZ3UtOXUayZK/vCP1YsH1agmEqXuKsCu6Fc+K8VzReOHxLUkmXn\nFYM+tixGahXcjEOi/aNNTWitEB6OemRM1UeLJFzRcfyXiqzHpHCIZwBpTUAsmzcG\nQiVDkMTKubwo/m+PVXburX2CGibUydctgbrYIc7EJvyx/cpRiPZXo1PhHQWdu4Y1\nSOaC66WLsP/wqvtHo58JQ6EN/gjSsbAgGGVkZ1xMo66nR+pLpR27coS7o03xCks6\nDY/0mukCgYEAuLIGgBnqoh7YsOBLd/Bc1UTfDMxJhNseo+hZemtkSXz2Jn51322F\nZw/FVN4ArXgluH+XsOhvG/MFFpojwZSrb0Qq5b1MRdo9qycq8lGqNtlN1WHqosDQ\nzW29kpL0tlRrSDpww3wRESsN9rH5XIrJ1b3ZXuO7asR+KBVQMy/+NcUCgYEA6MSC\nc+fywltKPgmPl5j0DPoDe5SXE/6JQy7w/vVGrGfWGf/zEJmhzS2R+CcfTTEqaT0T\nYw8+XbFgKAqsxwtE9MUXLTVLI3sSUyE4g7blCYscOqhZ8ItCUKDXWkSpt++rG0Um\n1+cEJP/0oCazG6MWqvBC4NpQ1nzh46QpjWqMwokCgYAKDLXJ1p8rvx3vUeUJW6zR\ndfPlEGCXuAyMwqHLxXgpf4EtSwhC5gSyPOtx2LqUtcrnpRmt6JfTH4ARYMW9TMef\nQEhNQ+WYj213mKP/l235mg1gJPnNbUxvQR9lkFV8bk+AGJ32JRQQqRUTbU+yN2MQ\nHEptnVqfTp3GtJIultfwOQKBgG+RyYmu8wBP650izg33BXu21raEeYne5oIqXN+I\nR5DZ0JjzwtkBGroTDrVoYyuH1nFNEh7YLqeQHqvyufBKKYo9cid8NQDTu+vWr5UK\ntGvHnwdKrJmM1oN5JOAiq0r7+QMAOWchVy449VNSWWV03aeftB685iR5BXkstbIQ\nEVopAoGAfcGBTAhmceK/4Q83H/FXBWy0PAa1kZGg/q8+Z0KY76AqyxOVl0/CU/rB\n3tO3sKhaMTHPME/MiQjQQGoaK1JgPY6JHYvly2KomrJ8QTugqNGyMzdVJkXAK2AM\nGAwC8ivAkHf8CHrHa1W7l8t2IqBjW1aRt7mOW92nfG88Hck0Mbo=\n-----END RSA PRIVATE KEY-----\n'
MASTER2_PUB_KEY = '\n-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAp+8cTxguO6Vg+YO92VfH\ngNld3Zy8aM3JbZvpJcjTnis+YFJ7Zlkcc647yPRRwY9nYBNywahnt5kIeuT1rTvT\nsMBZWvmUoEVUj1Xg8XXQkBvb9OzyGqy/G/p8KDDpzMP/U+XCnUeHiXTZrgnqgBIc\n2cKeCVvWFqDi0GRFGzyaXLaX3PPmM7DJ0MIPL1qgmcDq6+7Ze0gJ9SrDYFAeLmbu\nT1OqDfufXWQl/82JXeiwU2cOpqWq7n5fvPOWim7l1tzQ+dSiMRRm0xa6uNexCJww\n3oJSwvMbAmgzvOhqqhlqv+K7u0u7FrFFojESsL36Gq4GBrISnvu2tk7u4GGNTYYQ\nbQIDAQAB\n-----END PUBLIC KEY-----\n'
MASTER_SIGNING_PRIV = '\n-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAtieqrBMTM0MSIbhPKkDcozHqyXKyL/+bXYYw+iVPsns7c7bJ\nzBqenLQlWoRVyrVyBFrrwQSrKu/0Mqn3l639iOGPlUoR3I7aZKIpyEdDkqd3xGIC\ne+BtNNDqhUai67L63hEdG+iYAchi8UZw3LZGtcGpJ3FkBH4cYFX9EOam2QjbD7WY\nEO7m1+j6XEYIOTCmAP9dGAvBbU0Jblc+wYxG3qNr+2dBWsK76QXWEqib2VSOGP+z\ngjJa8tqY7PXXdOJpalQXNphmD/4o4pHKR4Euy0yL/1oMkpacmrV61LWB8Trnx9nS\n9gdVrUteQF/cL1KAGwOsdVmiLpHfvqLLRqSAAQIDAQABAoIBABjB+HEN4Kixf4fk\nwKHKEhL+SF6b/7sFX00NXZ/KLXRhSnnWSMQ8g/1hgMg2P2DfW4FbCDsCUu9xkLvI\nHTZY+CJAIh9U42uaYPWXkt09TmJi76TZ+2Nx4/XvRUjbCm7Fs1I2ekHeUbbAUS5g\n+BsPjTnL+h05zLHNoDa5yT0gVGIgFsQcX/w38arZCe8Rjp9le7PXUB5IIqASsDiw\nt8zJvdyWToeXd0WswCHTQu5coHvKo5MCjIZZ1Ink1yJcCCc3rKDc+q3jB2z9T9oW\ncUsKzJ4VuleiYj1eRxFITBmXbjKrb/GPRRUkeqCQbs68Hyj2d3UtOFDPeF4vng/3\njGsHPq8CgYEA0AHAbwykVC6NMa37BTvEqcKoxbjTtErxR+yczlmVDfma9vkwtZvx\nFJdbS/+WGA/ucDby5x5b2T5k1J9ueMR86xukb+HnyS0WKsZ94Ie8WnJAcbp+38M6\n7LD0u74Cgk93oagDAzUHqdLq9cXxv/ppBpxVB1Uvu8DfVMHj+wt6ie8CgYEA4C7u\nu+6b8EmbGqEdtlPpScKG0WFstJEDGXRARDCRiVP2w6wm25v8UssCPvWcwf8U1Hoq\nlhMY+H6a5dnRRiNYql1MGQAsqMi7VeJNYb0B1uxi7X8MPM+SvXoAglX7wm1z0cVy\nO4CE5sEKbBg6aQabx1x9tzdrm80SKuSsLc5HRQ8CgYEAp/mCKSuQWNru8ruJBwTp\nIB4upN1JOUN77ZVKW+lD0XFMjz1U9JPl77b65ziTQQM8jioRpkqB6cHVM088qxIh\nvssn06Iex/s893YrmPKETJYPLMhqRNEn+JQ+To53ADykY0uGg0SD18SYMbmULHBP\n+CKvF6jXT0vGDnA1ZzoxzskCgYEA2nQhYrRS9EVlhP93KpJ+A8gxA5tCCHo+YPFt\nJoWFbCKLlYUNoHZR3IPCPoOsK0Zbj+kz0mXtsUf9vPkR+py669haLQqEejyQgFIz\nQYiiYEKc6/0feapzvXtDP751w7JQaBtVAzJrT0jQ1SCO2oT8C7rPLlgs3fdpOq72\nMPSPcnUCgYBWHm6bn4HvaoUSr0v2hyD9fHZS/wDTnlXVe5c1XXgyKlJemo5dvycf\nHUCmN/xIuO6AsiMdqIzv+arNJdboz+O+bNtS43LkTJfEH3xj2/DdUogdvOgG/iPM\nu9KBT1h+euws7PqC5qt4vqLwCTTCZXmUS8Riv+62RCC3kZ5AbpT3ZA==\n-----END RSA PRIVATE KEY-----\n'
MASTER_SIGNING_PUB = '\n-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAtieqrBMTM0MSIbhPKkDc\nozHqyXKyL/+bXYYw+iVPsns7c7bJzBqenLQlWoRVyrVyBFrrwQSrKu/0Mqn3l639\niOGPlUoR3I7aZKIpyEdDkqd3xGICe+BtNNDqhUai67L63hEdG+iYAchi8UZw3LZG\ntcGpJ3FkBH4cYFX9EOam2QjbD7WYEO7m1+j6XEYIOTCmAP9dGAvBbU0Jblc+wYxG\n3qNr+2dBWsK76QXWEqib2VSOGP+zgjJa8tqY7PXXdOJpalQXNphmD/4o4pHKR4Eu\ny0yL/1oMkpacmrV61LWB8Trnx9nS9gdVrUteQF/cL1KAGwOsdVmiLpHfvqLLRqSA\nAQIDAQAB\n-----END PUBLIC KEY-----\n'
MINION_PRIV_KEY = '\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAsT6TwnlI0L7urjXu6D5E11tFJ/NglQ45jW/WN9tAUNvphq6Q\ncjJCd/aWmdqlqe7ix8y9M/8rgwghRQsnPXblVBvPwFcUEXhMRnOGzqbq/0zyQX01\nKecT0plBhlDt2lTyCLU6E4XCqyLbPfOxgXzsVqM0/TnzRtpVvGNy+5N4eFGylrjb\ncJhPxKt2G9TDOCM/hYacDs5RVIYQQmcYb8LJq7G3++FfWpYRDaxdKoHNFDspEynd\njzr67hgThnwzc388OKNJx/7B2atwPTunPb3YBjgwDyRO/01OKK4gUHdw5KoctFgp\nkDCDjwjemlyXV+MYODRTIdtOlAP83ZkntEuLoQIDAQABAoIBAAJOKNtvFGfF2l9H\nS4CXZSUGU0a+JaCkR+wmnjsPwPn/dXDpAe8nGpidpNicPWqRm6WABjeQHaxda+fB\nlpSrRtEdo3zoi2957xQJ5wddDtI1pmXJQrdbm0H/K39oIg/Xtv/IZT769TM6OtVg\npaUxG/aftmeGXDtGfIL8w1jkuPABRBLOakWQA9uVdeG19KTU0Ag8ilpJdEX64uFJ\nW75bpVjT+KO/6aV1inuCntQSP097aYvUWajRwuiYVJOxoBZHme3IObcE6mdnYXeQ\nwblyWBpJUHrOS4MP4HCODV2pHKZ2rr7Nwhh8lMNw/eY9OP0ifz2AcAqe3sUMQOKP\nT0qRC6ECgYEAyeU5JvUPOpxXvvChYh6gJ8pYTIh1ueDP0O5e4t3vhz6lfy9DKtRN\nROJLUorHvw/yVXMR72nT07a0z2VswcrUSw8ov3sI53F0NkLGEafQ35lVhTGs4vTl\nCFoQCuAKPsxeUl4AIbfbpkDsLGQqzW1diFArK7YeQkpGuGaGodXl480CgYEA4L40\nx5cUXnAhTPsybo7sbcpiwFHoGblmdkvpYvHA2QxtNSi2iHHdqGo8qP1YsZjKQn58\n371NhtqidrJ6i/8EBFP1dy+y/jr9qYlZNNGcQeBi+lshrEOIf1ct56KePG79s8lm\nDmD1OY8tO2R37+Py46Nq1n6viT/ST4NjLQI3GyUCgYEAiOswSDA3ZLs0cqRD/gPg\n/zsliLmehTFmHj4aEWcLkz+0Ar3tojUaNdX12QOPFQ7efH6uMhwl8NVeZ6xUBlTk\nhgbAzqLE1hjGBCpiowSZDZqyOcMHiV8ll/VkHcv0hsQYT2m6UyOaDXTH9g70TB6Y\nKOKddGZsvO4cad/1+/jQkB0CgYAzDEEkzLY9tS57M9uCrUgasAu6L2CO50PUvu1m\nIg9xvZbYqkS7vVFhva/FmrYYsOHQNLbcgz0m0mZwm52mSuh4qzFoPxdjE7cmWSJA\nExRxCiyxPR3q6PQKKJ0urgtPIs7RlX9u6KsKxfC6OtnbTWWQO0A7NE9e13ZHxUoz\noPsvWQKBgCa0+Fb2lzUeiQz9bV1CBkWneDZUXuZHmabAZomokX+h/bq+GcJFzZjW\n3kAHwYkIy9IAy3SyO/6CP0V3vAye1p+XbotiwsQ/XZnr0pflSQL3J1l1CyN3aopg\nNiv7k/zBn15B72aK73R/CpUSk9W/eJGqk1NcNwf8hJHsboRYx6BR\n-----END RSA PRIVATE KEY-----\n'
MINION_PUB_KEY = '\n-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAsT6TwnlI0L7urjXu6D5E\n11tFJ/NglQ45jW/WN9tAUNvphq6QcjJCd/aWmdqlqe7ix8y9M/8rgwghRQsnPXbl\nVBvPwFcUEXhMRnOGzqbq/0zyQX01KecT0plBhlDt2lTyCLU6E4XCqyLbPfOxgXzs\nVqM0/TnzRtpVvGNy+5N4eFGylrjbcJhPxKt2G9TDOCM/hYacDs5RVIYQQmcYb8LJ\nq7G3++FfWpYRDaxdKoHNFDspEyndjzr67hgThnwzc388OKNJx/7B2atwPTunPb3Y\nBjgwDyRO/01OKK4gUHdw5KoctFgpkDCDjwjemlyXV+MYODRTIdtOlAP83ZkntEuL\noQIDAQAB\n-----END PUBLIC KEY-----\n'
AES_KEY = '8wxWlOaMMQ4d3yT74LL4+hGrGTf65w8VgrcNjLJeLRQ2Q6zMa8ItY2EQUgMKKDb7JY+RnPUxbB0='

@pytest.fixture
def pki_dir(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    _pki_dir = tmp_path / 'pki'
    _pki_dir.mkdir()
    madir = _pki_dir / 'master'
    madir.mkdir()
    mapriv = madir / 'master.pem'
    mapriv.write_text(MASTER_PRIV_KEY.strip())
    mapub = madir / 'master.pub'
    mapub.write_text(MASTER_PUB_KEY.strip())
    maspriv = madir / 'master_sign.pem'
    maspriv.write_text(MASTER_SIGNING_PRIV.strip())
    maspub = madir / 'master_sign.pub'
    maspub.write_text(MASTER_SIGNING_PUB.strip())
    misdir = madir / 'minions'
    misdir.mkdir()
    misdir.joinpath('minion').write_text(MINION_PUB_KEY.strip())
    for sdir in ['minions_autosign', 'minions_denied', 'minions_pre', 'minions_rejected']:
        madir.joinpath(sdir).mkdir()
    midir = _pki_dir / 'minion'
    midir.mkdir()
    mipub = midir / 'minion.pub'
    mipub.write_text(MINION_PUB_KEY.strip())
    mipriv = midir / 'minion.pem'
    mipriv.write_text(MINION_PRIV_KEY.strip())
    mimapriv = midir / 'minion_master.pub'
    mimapriv.write_text(MASTER_PUB_KEY.strip())
    mimaspriv = midir / 'master_sign.pub'
    mimaspriv.write_text(MASTER_SIGNING_PUB.strip())
    yield _pki_dir

def test_master_uri():
    if False:
        i = 10
        return i + 15
    '\n    test _get_master_uri method\n    '
    m_ip = '127.0.0.1'
    m_port = 4505
    s_ip = '111.1.0.1'
    s_port = 4058
    m_ip6 = '1234:5678::9abc'
    s_ip6 = '1234:5678::1:9abc'
    with patch('salt.transport.zeromq.LIBZMQ_VERSION_INFO', (4, 1, 6)), patch('salt.transport.zeromq.ZMQ_VERSION_INFO', (16, 0, 1)):
        assert salt.transport.zeromq._get_master_uri(master_ip=m_ip, master_port=m_port, source_ip=s_ip, source_port=s_port) == f'tcp://{s_ip}:{s_port};{m_ip}:{m_port}'
        assert salt.transport.zeromq._get_master_uri(master_ip=m_ip6, master_port=m_port, source_ip=s_ip6, source_port=s_port) == f'tcp://[{s_ip6}]:{s_port};[{m_ip6}]:{m_port}'
        assert salt.transport.zeromq._get_master_uri(master_ip=m_ip, master_port=m_port) == f'tcp://{m_ip}:{m_port}'
        assert salt.transport.zeromq._get_master_uri(master_ip=m_ip6, master_port=m_port) == f'tcp://[{m_ip6}]:{m_port}'
        assert salt.transport.zeromq._get_master_uri(master_ip=m_ip, master_port=m_port, source_ip=s_ip) == f'tcp://{s_ip}:0;{m_ip}:{m_port}'
        assert salt.transport.zeromq._get_master_uri(master_ip=m_ip6, master_port=m_port, source_ip=s_ip6) == f'tcp://[{s_ip6}]:0;[{m_ip6}]:{m_port}'
        assert salt.transport.zeromq._get_master_uri(master_ip=m_ip, master_port=m_port, source_port=s_port) == f'tcp://0.0.0.0:{s_port};{m_ip}:{m_port}'

def test_clear_req_channel_master_uri_override(temp_salt_minion, temp_salt_master):
    if False:
        i = 10
        return i + 15
    '\n    ensure master_uri kwarg is respected\n    '
    opts = temp_salt_minion.config.copy()
    opts.update({'id': 'root', 'transport': 'zeromq', 'auth_tries': 1, 'auth_timeout': 5, 'master_ip': '127.0.0.1', 'master_port': temp_salt_master.config['ret_port'], 'master_uri': 'tcp://127.0.0.1:{}'.format(temp_salt_master.config['ret_port'])})
    master_uri = 'tcp://{master_ip}:{master_port}'.format(master_ip='localhost', master_port=opts['master_port'])
    with salt.channel.client.ReqChannel.factory(opts, master_uri=master_uri) as channel:
        assert '127.0.0.1' in channel.transport.master_uri

def run_loop_in_thread(loop, evt):
    if False:
        while True:
            i = 10
    '\n    Run the provided loop until an event is set\n    '
    asyncio.set_event_loop(loop.asyncio_loop)

    async def stopper():
        await asyncio.sleep(0.1)
        while True:
            if not evt.is_set():
                loop.stop()
                break
            await asyncio.sleep(0.3)
    loop.add_callback(evt.set)
    loop.add_callback(stopper)
    try:
        loop.start()
    finally:
        loop.close()

class MockSaltMinionMaster:
    mock = MagicMock()

    def __init__(self, temp_salt_minion, temp_salt_master):
        if False:
            print('Hello World!')
        SMaster.secrets['aes'] = {'secret': multiprocessing.Array(ctypes.c_char, salt.utils.stringutils.to_bytes(salt.crypt.Crypticle.generate_key_string())), 'reload': salt.crypt.Crypticle.generate_key_string}
        self.process_manager = salt.utils.process.ProcessManager(name='ReqServer_ProcessManager')
        master_opts = temp_salt_master.config.copy()
        master_opts.update({'transport': 'zeromq'})
        self.server_channel = salt.channel.server.ReqServerChannel.factory(master_opts)
        self.server_channel.pre_fork(self.process_manager)
        self.io_loop = tornado.ioloop.IOLoop(make_current=False)
        self.evt = threading.Event()
        self.server_channel.post_fork(self._handle_payload, io_loop=self.io_loop)
        self.server_thread = threading.Thread(target=run_loop_in_thread, args=(self.io_loop, self.evt))
        self.server_thread.start()
        minion_opts = temp_salt_minion.config.copy()
        minion_opts.update({'master_ip': '127.0.0.1', 'transport': 'zeromq'})
        self.channel = salt.channel.client.ReqChannel.factory(minion_opts, crypt='clear')

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.channel.__enter__()
        self.evt.wait()
        return self

    def __exit__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.channel.__exit__(*args, **kwargs)
        del self.channel
        self.server_channel.close()
        self.process_manager.stop_restarting()
        self.process_manager.kill_children()
        self.evt.clear()
        self.server_thread.join()
        time.sleep(2)
        SMaster.secrets.pop('aes')
        del self.server_channel
        del self.io_loop
        del self.process_manager
        del self.server_thread

    @classmethod
    @tornado.gen.coroutine
    def _handle_payload(cls, payload):
        if False:
            print('Hello World!')
        '\n        TODO: something besides echo\n        '
        cls.mock._handle_payload_hook()
        raise tornado.gen.Return((payload, {'fun': 'send_clear'}))

@pytest.mark.parametrize('message', ['', [], ()])
def test_badload(temp_salt_minion, temp_salt_master, message):
    if False:
        return 10
    '\n    Test a variety of bad requests, make sure that we get some sort of error\n    '
    with MockSaltMinionMaster(temp_salt_minion, temp_salt_master) as minion_master:
        ret = minion_master.channel.send(message, timeout=5, tries=1)
        assert ret == 'payload and load must be a dict'

def test_payload_handling_exception(temp_salt_minion, temp_salt_master):
    if False:
        for i in range(10):
            print('nop')
    '\n    test of getting exception on payload handling\n    '
    with MockSaltMinionMaster(temp_salt_minion, temp_salt_master) as minion_master:
        with patch.object(minion_master.mock, '_handle_payload_hook') as _mock:
            _mock.side_effect = Exception()
            ret = minion_master.channel.send({}, timeout=15, tries=1)
            assert ret == 'Some exception handling minion payload'

def test_serverside_exception(temp_salt_minion, temp_salt_master):
    if False:
        i = 10
        return i + 15
    '\n    test of getting server side exception on payload handling\n    '
    with MockSaltMinionMaster(temp_salt_minion, temp_salt_master) as minion_master:
        with patch.object(minion_master.mock, '_handle_payload_hook') as _mock:
            _mock.side_effect = tornado.gen.Return(({}, {'fun': 'madeup-fun'}))
            ret = minion_master.channel.send({}, timeout=5, tries=1)
            assert ret == 'Server-side exception handling payload'

def test_req_server_chan_encrypt_v2(master_opts, pki_dir):
    if False:
        while True:
            i = 10
    loop = tornado.ioloop.IOLoop.current()
    master_opts.update({'worker_threads': 1, 'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'zmq_monitor': False, 'mworker_queue_niceness': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('master')), 'id': 'minion', '__role': 'minion', 'keysize': 4096})
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    dictkey = 'pillar'
    nonce = 'abcdefg'
    pillar_data = {'pillar1': 'meh'}
    try:
        ret = server._encrypt_private(pillar_data, dictkey, 'minion', nonce)
        assert 'key' in ret
        assert dictkey in ret
        key = salt.crypt.get_rsa_key(str(pki_dir.joinpath('minion', 'minion.pem')), None)
        if HAS_M2:
            aes = key.private_decrypt(ret['key'], RSA.pkcs1_oaep_padding)
        else:
            cipher = PKCS1_OAEP.new(key)
            aes = cipher.decrypt(ret['key'])
        pcrypt = salt.crypt.Crypticle(master_opts, aes)
        signed_msg = pcrypt.loads(ret[dictkey])
        assert 'sig' in signed_msg
        assert 'data' in signed_msg
        data = salt.payload.loads(signed_msg['data'])
        assert 'key' in data
        assert data['key'] == ret['key']
        assert 'key' in data
        assert data['nonce'] == nonce
        assert 'pillar' in data
        assert data['pillar'] == pillar_data
    finally:
        server.close()

def test_req_server_chan_encrypt_v1(master_opts, pki_dir):
    if False:
        print('Hello World!')
    loop = tornado.ioloop.IOLoop.current()
    master_opts.update({'worker_threads': 1, 'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'zmq_monitor': False, 'mworker_queue_niceness': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('master')), 'id': 'minion', '__role': 'minion', 'keysize': 4096})
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    dictkey = 'pillar'
    nonce = 'abcdefg'
    pillar_data = {'pillar1': 'meh'}
    try:
        ret = server._encrypt_private(pillar_data, dictkey, 'minion', sign_messages=False)
        assert 'key' in ret
        assert dictkey in ret
        key = salt.crypt.get_rsa_key(str(pki_dir.joinpath('minion', 'minion.pem')), None)
        if HAS_M2:
            aes = key.private_decrypt(ret['key'], RSA.pkcs1_oaep_padding)
        else:
            cipher = PKCS1_OAEP.new(key)
            aes = cipher.decrypt(ret['key'])
        pcrypt = salt.crypt.Crypticle(master_opts, aes)
        data = pcrypt.loads(ret[dictkey])
        assert data == pillar_data
    finally:
        server.close()

def test_req_chan_decode_data_dict_entry_v1(minion_opts, master_opts, pki_dir):
    if False:
        while True:
            i = 10
    mockloop = MagicMock()
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'acceptance_wait_time': 3, 'acceptance_wait_time_max': 3})
    master_opts.update(pki_dir=str(pki_dir.joinpath('master')))
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    client = salt.channel.client.ReqChannel.factory(minion_opts, io_loop=mockloop)
    try:
        dictkey = 'pillar'
        target = 'minion'
        pillar_data = {'pillar1': 'meh'}
        ret = server._encrypt_private(pillar_data, dictkey, target, sign_messages=False)
        key = client.auth.get_keys()
        if HAS_M2:
            aes = key.private_decrypt(ret['key'], RSA.pkcs1_oaep_padding)
        else:
            cipher = PKCS1_OAEP.new(key)
            aes = cipher.decrypt(ret['key'])
        pcrypt = salt.crypt.Crypticle(client.opts, aes)
        ret_pillar_data = pcrypt.loads(ret[dictkey])
        assert ret_pillar_data == pillar_data
    finally:
        client.close()
        server.close()

async def test_req_chan_decode_data_dict_entry_v2(minion_opts, master_opts, pki_dir):
    mockloop = MagicMock()
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'acceptance_wait_time': 3, 'acceptance_wait_time_max': 3})
    master_opts.update(pki_dir=str(pki_dir.joinpath('master')))
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    client = salt.channel.client.AsyncReqChannel.factory(minion_opts, io_loop=mockloop)
    dictkey = 'pillar'
    target = 'minion'
    pillar_data = {'pillar1': 'meh'}
    auth = client.auth
    auth._crypticle = salt.crypt.Crypticle(minion_opts, AES_KEY)
    client.auth = MagicMock()
    client.auth.mpub = auth.mpub
    client.auth.authenticated = True
    client.auth.get_keys = auth.get_keys
    client.auth.crypticle.dumps = auth.crypticle.dumps
    client.auth.crypticle.loads = auth.crypticle.loads
    real_transport = client.transport
    client.transport = MagicMock()
    real_transport.close()

    @tornado.gen.coroutine
    def mocksend(msg, timeout=60, tries=3):
        if False:
            for i in range(10):
                print('nop')
        client.transport.msg = msg
        load = client.auth.crypticle.loads(msg['load'])
        ret = server._encrypt_private(pillar_data, dictkey, target, nonce=load['nonce'], sign_messages=True)
        raise tornado.gen.Return(ret)
    client.transport.send = mocksend
    load = {'id': target, 'grains': {}, 'saltenv': 'base', 'pillarenv': 'base', 'pillar_override': True, 'extra_minion_data': {}, 'ver': '2', 'cmd': '_pillar'}
    try:
        ret = await client.crypted_transfer_decode_dictentry(load, dictkey='pillar')
        assert 'version' in client.transport.msg
        assert client.transport.msg['version'] == 2
        assert ret == {'pillar1': 'meh'}
    finally:
        client.close()
        server.close()

async def test_req_chan_decode_data_dict_entry_v2_bad_nonce(minion_opts, master_opts, pki_dir):
    mockloop = MagicMock()
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'acceptance_wait_time': 3, 'acceptance_wait_time_max': 3})
    master_opts.update(pki_dir=str(pki_dir.joinpath('master')))
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    client = salt.channel.client.AsyncReqChannel.factory(minion_opts, io_loop=mockloop)
    dictkey = 'pillar'
    badnonce = 'abcdefg'
    target = 'minion'
    pillar_data = {'pillar1': 'meh'}
    auth = client.auth
    auth._crypticle = salt.crypt.Crypticle(minion_opts, AES_KEY)
    client.auth = MagicMock()
    client.auth.mpub = auth.mpub
    client.auth.authenticated = True
    client.auth.get_keys = auth.get_keys
    client.auth.crypticle.dumps = auth.crypticle.dumps
    client.auth.crypticle.loads = auth.crypticle.loads
    real_transport = client.transport
    client.transport = MagicMock()
    real_transport.close()
    ret = server._encrypt_private(pillar_data, dictkey, target, nonce=badnonce, sign_messages=True)

    @tornado.gen.coroutine
    def mocksend(msg, timeout=60, tries=3):
        if False:
            print('Hello World!')
        client.transport.msg = msg
        raise tornado.gen.Return(ret)
    client.transport.send = mocksend
    load = {'id': target, 'grains': {}, 'saltenv': 'base', 'pillarenv': 'base', 'pillar_override': True, 'extra_minion_data': {}, 'ver': '2', 'cmd': '_pillar'}
    try:
        with pytest.raises(salt.crypt.AuthenticationError) as excinfo:
            ret = await client.crypted_transfer_decode_dictentry(load, dictkey='pillar')
        assert 'Pillar nonce verification failed.' == excinfo.value.message
    finally:
        client.close()
        server.close()

async def test_req_chan_decode_data_dict_entry_v2_bad_signature(minion_opts, master_opts, pki_dir):
    mockloop = MagicMock()
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'acceptance_wait_time': 3, 'acceptance_wait_time_max': 3})
    master_opts.update(pki_dir=str(pki_dir.joinpath('master')))
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    client = salt.channel.client.AsyncReqChannel.factory(minion_opts, io_loop=mockloop)
    dictkey = 'pillar'
    badnonce = 'abcdefg'
    target = 'minion'
    pillar_data = {'pillar1': 'meh'}
    auth = client.auth
    auth._crypticle = salt.crypt.Crypticle(minion_opts, AES_KEY)
    client.auth = MagicMock()
    client.auth.mpub = auth.mpub
    client.auth.authenticated = True
    client.auth.get_keys = auth.get_keys
    client.auth.crypticle.dumps = auth.crypticle.dumps
    client.auth.crypticle.loads = auth.crypticle.loads
    real_transport = client.transport
    client.transport = MagicMock()
    real_transport.close()

    @tornado.gen.coroutine
    def mocksend(msg, timeout=60, tries=3):
        if False:
            while True:
                i = 10
        client.transport.msg = msg
        load = client.auth.crypticle.loads(msg['load'])
        ret = server._encrypt_private(pillar_data, dictkey, target, nonce=load['nonce'], sign_messages=True)
        key = client.auth.get_keys()
        if HAS_M2:
            aes = key.private_decrypt(ret['key'], RSA.pkcs1_oaep_padding)
        else:
            cipher = PKCS1_OAEP.new(key)
            aes = cipher.decrypt(ret['key'])
        pcrypt = salt.crypt.Crypticle(client.opts, aes)
        signed_msg = pcrypt.loads(ret[dictkey])
        data = salt.payload.loads(signed_msg['data'])
        data['pillar'] = {'pillar1': 'bar'}
        signed_msg['data'] = salt.payload.dumps(data)
        ret[dictkey] = pcrypt.dumps(signed_msg)
        raise tornado.gen.Return(ret)
    client.transport.send = mocksend
    load = {'id': target, 'grains': {}, 'saltenv': 'base', 'pillarenv': 'base', 'pillar_override': True, 'extra_minion_data': {}, 'ver': '2', 'cmd': '_pillar'}
    try:
        with pytest.raises(salt.crypt.AuthenticationError) as excinfo:
            ret = await client.crypted_transfer_decode_dictentry(load, dictkey='pillar')
        assert 'Pillar payload signature failed to validate.' == excinfo.value.message
    finally:
        client.close()
        server.close()

async def test_req_chan_decode_data_dict_entry_v2_bad_key(minion_opts, master_opts, pki_dir):
    mockloop = MagicMock()
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'acceptance_wait_time': 3, 'acceptance_wait_time_max': 3})
    master_opts.update(pki_dir=str(pki_dir.joinpath('master')))
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    client = salt.channel.client.AsyncReqChannel.factory(minion_opts, io_loop=mockloop)
    dictkey = 'pillar'
    badnonce = 'abcdefg'
    target = 'minion'
    pillar_data = {'pillar1': 'meh'}
    auth = client.auth
    auth._crypticle = salt.crypt.Crypticle(minion_opts, AES_KEY)
    client.auth = MagicMock()
    client.auth.mpub = auth.mpub
    client.auth.authenticated = True
    client.auth.get_keys = auth.get_keys
    client.auth.crypticle.dumps = auth.crypticle.dumps
    client.auth.crypticle.loads = auth.crypticle.loads
    real_transport = client.transport
    client.transport = MagicMock()
    real_transport.close()

    @tornado.gen.coroutine
    def mocksend(msg, timeout=60, tries=3):
        if False:
            for i in range(10):
                print('nop')
        client.transport.msg = msg
        load = client.auth.crypticle.loads(msg['load'])
        ret = server._encrypt_private(pillar_data, dictkey, target, nonce=load['nonce'], sign_messages=True)
        key = client.auth.get_keys()
        if HAS_M2:
            aes = key.private_decrypt(ret['key'], RSA.pkcs1_oaep_padding)
        else:
            cipher = PKCS1_OAEP.new(key)
            aes = cipher.decrypt(ret['key'])
        pcrypt = salt.crypt.Crypticle(client.opts, aes)
        signed_msg = pcrypt.loads(ret[dictkey])
        key = salt.crypt.Crypticle.generate_key_string()
        pcrypt = salt.crypt.Crypticle(minion_opts, key)
        pubfn = os.path.join(master_opts['pki_dir'], 'minions', 'minion')
        pub = salt.crypt.get_rsa_pub_key(pubfn)
        ret[dictkey] = pcrypt.dumps(signed_msg)
        key = salt.utils.stringutils.to_bytes(key)
        if HAS_M2:
            ret['key'] = pub.public_encrypt(key, RSA.pkcs1_oaep_padding)
        else:
            cipher = PKCS1_OAEP.new(pub)
            ret['key'] = cipher.encrypt(key)
        raise tornado.gen.Return(ret)
    client.transport.send = mocksend
    load = {'id': target, 'grains': {}, 'saltenv': 'base', 'pillarenv': 'base', 'pillar_override': True, 'extra_minion_data': {}, 'ver': '2', 'cmd': '_pillar'}
    try:
        with pytest.raises(salt.crypt.AuthenticationError) as excinfo:
            await client.crypted_transfer_decode_dictentry(load, dictkey='pillar')
        assert 'Key verification failed.' == excinfo.value.message
    finally:
        client.close()
        server.close()

async def test_req_serv_auth_v1(minion_opts, master_opts, pki_dir):
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'max_minions': 0, 'auto_accept': False, 'open_mode': False, 'key_pass': None, 'master_sign_pubkey': False, 'publish_port': 4505, 'auth_mode': 1})
    SMaster.secrets['aes'] = {'secret': multiprocessing.Array(ctypes.c_char, salt.utils.stringutils.to_bytes(salt.crypt.Crypticle.generate_key_string())), 'reload': salt.crypt.Crypticle.generate_key_string}
    master_opts.update(pki_dir=str(pki_dir.joinpath('master')))
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    server.auto_key = salt.daemons.masterapi.AutoKey(server.opts)
    server.cache_cli = False
    server.master_key = salt.crypt.MasterKeys(server.opts)
    pub = salt.crypt.get_rsa_pub_key(str(pki_dir.joinpath('minion', 'minion.pub')))
    token = salt.utils.stringutils.to_bytes(salt.crypt.Crypticle.generate_key_string())
    nonce = uuid.uuid4().hex
    with salt.utils.files.fopen(str(pki_dir.joinpath('minion', 'minion.pub')), 'r') as fp:
        pub_key = fp.read()
    load = {'cmd': '_auth', 'id': 'minion', 'token': token, 'pub': pub_key}
    try:
        ret = server._auth(load, sign_messages=False)
        assert 'load' not in ret
    finally:
        server.close()

async def test_req_serv_auth_v2(minion_opts, master_opts, pki_dir):
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'max_minions': 0, 'auto_accept': False, 'open_mode': False, 'key_pass': None, 'master_sign_pubkey': False, 'publish_port': 4505, 'auth_mode': 1})
    SMaster.secrets['aes'] = {'secret': multiprocessing.Array(ctypes.c_char, salt.utils.stringutils.to_bytes(salt.crypt.Crypticle.generate_key_string())), 'reload': salt.crypt.Crypticle.generate_key_string}
    master_opts.update(pki_dir=str(pki_dir.joinpath('master')))
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    server.auto_key = salt.daemons.masterapi.AutoKey(server.opts)
    server.cache_cli = False
    server.master_key = salt.crypt.MasterKeys(server.opts)
    pub = salt.crypt.get_rsa_pub_key(str(pki_dir.joinpath('minion', 'minion.pub')))
    token = salt.utils.stringutils.to_bytes(salt.crypt.Crypticle.generate_key_string())
    nonce = uuid.uuid4().hex
    with salt.utils.files.fopen(str(pki_dir.joinpath('minion', 'minion.pub')), 'r') as fp:
        pub_key = fp.read()
    load = {'cmd': '_auth', 'id': 'minion', 'nonce': nonce, 'token': token, 'pub': pub_key}
    try:
        ret = server._auth(load, sign_messages=True)
        assert 'sig' in ret
        assert 'load' in ret
    finally:
        server.close()

async def test_req_chan_auth_v2(minion_opts, master_opts, pki_dir, io_loop):
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'max_minions': 0, 'auto_accept': False, 'open_mode': False, 'key_pass': None, 'publish_port': 4505, 'auth_mode': 1, 'acceptance_wait_time': 3, 'acceptance_wait_time_max': 3})
    SMaster.secrets['aes'] = {'secret': multiprocessing.Array(ctypes.c_char, salt.utils.stringutils.to_bytes(salt.crypt.Crypticle.generate_key_string())), 'reload': salt.crypt.Crypticle.generate_key_string}
    master_opts.update(pki_dir=str(pki_dir.joinpath('master')))
    master_opts['master_sign_pubkey'] = False
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    server.auto_key = salt.daemons.masterapi.AutoKey(server.opts)
    server.cache_cli = False
    server.master_key = salt.crypt.MasterKeys(server.opts)
    minion_opts['verify_master_pubkey_sign'] = False
    minion_opts['always_verify_signature'] = False
    client = salt.channel.client.AsyncReqChannel.factory(minion_opts, io_loop=io_loop)
    signin_payload = client.auth.minion_sign_in_payload()
    pload = client._package_load(signin_payload)
    try:
        assert 'version' in pload
        assert pload['version'] == 2
        ret = server._auth(pload['load'], sign_messages=True)
        assert 'sig' in ret
        ret = client.auth.handle_signin_response(signin_payload, ret)
        assert 'aes' in ret
        assert 'master_uri' in ret
        assert 'publish_port' in ret
    finally:
        client.close()
        server.close()

async def test_req_chan_auth_v2_with_master_signing(minion_opts, master_opts, pki_dir, io_loop):
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'max_minions': 0, 'auto_accept': False, 'open_mode': False, 'key_pass': None, 'publish_port': 4505, 'auth_mode': 1, 'acceptance_wait_time': 3, 'acceptance_wait_time_max': 3})
    SMaster.secrets['aes'] = {'secret': multiprocessing.Array(ctypes.c_char, salt.utils.stringutils.to_bytes(salt.crypt.Crypticle.generate_key_string())), 'reload': salt.crypt.Crypticle.generate_key_string}
    master_opts.update(pki_dir=str(pki_dir.joinpath('master')))
    master_opts['master_sign_pubkey'] = True
    master_opts['master_use_pubkey_signature'] = False
    master_opts['signing_key_pass'] = True
    master_opts['master_sign_key_name'] = 'master_sign'
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    server.auto_key = salt.daemons.masterapi.AutoKey(server.opts)
    server.cache_cli = False
    server.master_key = salt.crypt.MasterKeys(server.opts)
    minion_opts['verify_master_pubkey_sign'] = True
    minion_opts['always_verify_signature'] = True
    minion_opts['master_sign_key_name'] = 'master_sign'
    minion_opts['master'] = 'master'
    assert pki_dir.joinpath('minion', 'minion_master.pub').read_text() == pki_dir.joinpath('master', 'master.pub').read_text()
    client = salt.channel.client.AsyncReqChannel.factory(minion_opts, io_loop=io_loop)
    signin_payload = client.auth.minion_sign_in_payload()
    pload = client._package_load(signin_payload)
    assert 'version' in pload
    assert pload['version'] == 2
    server_reply = server._auth(pload['load'], sign_messages=True)
    assert 'enc' in server_reply
    assert server_reply['enc'] == 'clear'
    assert 'sig' in server_reply
    assert 'load' in server_reply
    ret = client.auth.handle_signin_response(signin_payload, server_reply)
    assert 'aes' in ret
    assert 'master_uri' in ret
    assert 'publish_port' in ret
    mapriv = pki_dir.joinpath('master', 'master.pem')
    mapriv.unlink()
    mapriv.write_text(MASTER2_PRIV_KEY.strip())
    mapub = pki_dir.joinpath('master', 'master.pub')
    mapub.unlink()
    mapub.write_text(MASTER2_PUB_KEY.strip())
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    server.auto_key = salt.daemons.masterapi.AutoKey(server.opts)
    server.cache_cli = False
    server.master_key = salt.crypt.MasterKeys(server.opts)
    signin_payload = client.auth.minion_sign_in_payload()
    pload = client._package_load(signin_payload)
    server_reply = server._auth(pload['load'], sign_messages=True)
    try:
        ret = client.auth.handle_signin_response(signin_payload, server_reply)
        assert 'aes' in ret
        assert 'master_uri' in ret
        assert 'publish_port' in ret
        assert pki_dir.joinpath('minion', 'minion_master.pub').read_text() == pki_dir.joinpath('master', 'master.pub').read_text()
    finally:
        client.close()
        server.close()

async def test_req_chan_auth_v2_new_minion_with_master_pub(minion_opts, master_opts, pki_dir, io_loop):
    pki_dir.joinpath('master', 'minions', 'minion').unlink()
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'max_minions': 0, 'auto_accept': False, 'open_mode': False, 'key_pass': None, 'publish_port': 4505, 'auth_mode': 1, 'acceptance_wait_time': 3, 'acceptance_wait_time_max': 3})
    SMaster.secrets['aes'] = {'secret': multiprocessing.Array(ctypes.c_char, salt.utils.stringutils.to_bytes(salt.crypt.Crypticle.generate_key_string())), 'reload': salt.crypt.Crypticle.generate_key_string}
    master_opts.update(pki_dir=str(pki_dir.joinpath('master')))
    master_opts['master_sign_pubkey'] = False
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    server.auto_key = salt.daemons.masterapi.AutoKey(server.opts)
    server.cache_cli = False
    server.master_key = salt.crypt.MasterKeys(server.opts)
    minion_opts['verify_master_pubkey_sign'] = False
    minion_opts['always_verify_signature'] = False
    client = salt.channel.client.AsyncReqChannel.factory(minion_opts, io_loop=io_loop)
    signin_payload = client.auth.minion_sign_in_payload()
    pload = client._package_load(signin_payload)
    try:
        assert 'version' in pload
        assert pload['version'] == 2
        ret = server._auth(pload['load'], sign_messages=True)
        assert 'sig' in ret
        ret = client.auth.handle_signin_response(signin_payload, ret)
        assert ret == 'retry'
    finally:
        client.close()
        server.close()

async def test_req_chan_auth_v2_new_minion_with_master_pub_bad_sig(minion_opts, master_opts, pki_dir, io_loop):
    pki_dir.joinpath('master', 'minions', 'minion').unlink()
    mapriv = pki_dir.joinpath('master', 'master.pem')
    mapriv.unlink()
    mapriv.write_text(MASTER2_PRIV_KEY.strip())
    mapub = pki_dir.joinpath('master', 'master.pub')
    mapub.unlink()
    mapub.write_text(MASTER2_PUB_KEY.strip())
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'max_minions': 0, 'auto_accept': False, 'open_mode': False, 'key_pass': None, 'publish_port': 4505, 'auth_mode': 1, 'acceptance_wait_time': 3, 'acceptance_wait_time_max': 3})
    SMaster.secrets['aes'] = {'secret': multiprocessing.Array(ctypes.c_char, salt.utils.stringutils.to_bytes(salt.crypt.Crypticle.generate_key_string())), 'reload': salt.crypt.Crypticle.generate_key_string}
    master_opts.update(pki_dir=str(pki_dir.joinpath('master')))
    master_opts['master_sign_pubkey'] = False
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    server.auto_key = salt.daemons.masterapi.AutoKey(server.opts)
    server.cache_cli = False
    server.master_key = salt.crypt.MasterKeys(server.opts)
    minion_opts['verify_master_pubkey_sign'] = False
    minion_opts['always_verify_signature'] = False
    client = salt.channel.client.AsyncReqChannel.factory(minion_opts, io_loop=io_loop)
    signin_payload = client.auth.minion_sign_in_payload()
    pload = client._package_load(signin_payload)
    try:
        assert 'version' in pload
        assert pload['version'] == 2
        ret = server._auth(pload['load'], sign_messages=True)
        assert 'sig' in ret
        with pytest.raises(salt.crypt.SaltClientError, match='Invalid signature'):
            ret = client.auth.handle_signin_response(signin_payload, ret)
    finally:
        client.close()
        server.close()

async def test_req_chan_auth_v2_new_minion_without_master_pub(minion_opts, master_opts, pki_dir, io_loop):
    pki_dir.joinpath('master', 'minions', 'minion').unlink()
    pki_dir.joinpath('minion', 'minion_master.pub').unlink()
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'max_minions': 0, 'auto_accept': False, 'open_mode': False, 'key_pass': None, 'publish_port': 4505, 'auth_mode': 1, 'acceptance_wait_time': 3, 'acceptance_wait_time_max': 3})
    SMaster.secrets['aes'] = {'secret': multiprocessing.Array(ctypes.c_char, salt.utils.stringutils.to_bytes(salt.crypt.Crypticle.generate_key_string())), 'reload': salt.crypt.Crypticle.generate_key_string}
    master_opts.update(pki_dir=str(pki_dir.joinpath('master')))
    master_opts['master_sign_pubkey'] = False
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    server.auto_key = salt.daemons.masterapi.AutoKey(server.opts)
    server.cache_cli = False
    server.master_key = salt.crypt.MasterKeys(server.opts)
    minion_opts['verify_master_pubkey_sign'] = False
    minion_opts['always_verify_signature'] = False
    client = salt.channel.client.AsyncReqChannel.factory(minion_opts, io_loop=io_loop)
    signin_payload = client.auth.minion_sign_in_payload()
    pload = client._package_load(signin_payload)
    try:
        assert 'version' in pload
        assert pload['version'] == 2
        ret = server._auth(pload['load'], sign_messages=True)
        assert 'sig' in ret
        ret = client.auth.handle_signin_response(signin_payload, ret)
        assert ret == 'retry'
    finally:
        client.close()
        server.close()

async def test_req_chan_bad_payload_to_decode(minion_opts, master_opts, pki_dir, io_loop):
    minion_opts.update({'master_uri': 'tcp://127.0.0.1:4506', 'interface': '127.0.0.1', 'ret_port': 4506, 'ipv6': False, 'sock_dir': '.', 'pki_dir': str(pki_dir.joinpath('minion')), 'id': 'minion', '__role': 'minion', 'keysize': 4096, 'max_minions': 0, 'auto_accept': False, 'open_mode': False, 'key_pass': None, 'publish_port': 4505, 'auth_mode': 1, 'acceptance_wait_time': 3, 'acceptance_wait_time_max': 3})
    SMaster.secrets['aes'] = {'secret': multiprocessing.Array(ctypes.c_char, salt.utils.stringutils.to_bytes(salt.crypt.Crypticle.generate_key_string())), 'reload': salt.crypt.Crypticle.generate_key_string}
    master_opts.update(dict(minion_opts, pki_dir=str(pki_dir.joinpath('master'))))
    master_opts['master_sign_pubkey'] = False
    server = salt.channel.server.ReqServerChannel.factory(master_opts)
    with pytest.raises(salt.exceptions.SaltDeserializationError):
        server._decode_payload(None)
    with pytest.raises(salt.exceptions.SaltDeserializationError):
        server._decode_payload({})
    with pytest.raises(salt.exceptions.SaltDeserializationError):
        server._decode_payload(12345)