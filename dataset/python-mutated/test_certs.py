import os
from datetime import datetime
from datetime import timezone
from pathlib import Path
import pytest
from cryptography import x509
from cryptography.x509 import NameOID
from ..conftest import skip_windows
from mitmproxy import certs

@pytest.fixture()
def tstore(tdata):
    if False:
        for i in range(10):
            print('nop')
    return certs.CertStore.from_store(tdata.path('mitmproxy/data/confdir'), 'mitmproxy', 2048)

class TestCertStore:

    def test_create_explicit(self, tmpdir):
        if False:
            return 10
        ca = certs.CertStore.from_store(str(tmpdir), 'test', 2048)
        assert ca.get_cert('foo', [])
        ca2 = certs.CertStore.from_store(str(tmpdir), 'test', 2048)
        assert ca2.get_cert('foo', [])
        assert ca.default_ca.serial == ca2.default_ca.serial

    def test_create_no_common_name(self, tstore):
        if False:
            print('Hello World!')
        assert tstore.get_cert(None, []).cert.cn is None

    def test_chain_file(self, tdata, tmp_path):
        if False:
            return 10
        cert = Path(tdata.path('mitmproxy/data/confdir/mitmproxy-ca.pem')).read_bytes()
        (tmp_path / 'mitmproxy-ca.pem').write_bytes(cert)
        ca = certs.CertStore.from_store(tmp_path, 'mitmproxy', 2048)
        assert ca.default_chain_file is None
        assert len(ca.default_chain_certs) == 1
        (tmp_path / 'mitmproxy-ca.pem').write_bytes(2 * cert)
        ca = certs.CertStore.from_store(tmp_path, 'mitmproxy', 2048)
        assert ca.default_chain_file == tmp_path / 'mitmproxy-ca.pem'
        assert len(ca.default_chain_certs) == 2

    def test_sans(self, tstore):
        if False:
            print('Hello World!')
        c1 = tstore.get_cert('foo.com', ['*.bar.com'])
        tstore.get_cert('foo.bar.com', [])
        c3 = tstore.get_cert('bar.com', [])
        assert not c1 == c3

    def test_sans_change(self, tstore):
        if False:
            return 10
        tstore.get_cert('foo.com', ['*.bar.com'])
        entry = tstore.get_cert('foo.bar.com', ['*.baz.com'])
        assert '*.baz.com' in entry.cert.altnames

    def test_expire(self, tstore):
        if False:
            while True:
                i = 10
        tstore.STORE_CAP = 3
        tstore.get_cert('one.com', [])
        tstore.get_cert('two.com', [])
        tstore.get_cert('three.com', [])
        assert ('one.com', ()) in tstore.certs
        assert ('two.com', ()) in tstore.certs
        assert ('three.com', ()) in tstore.certs
        tstore.get_cert('one.com', [])
        assert ('one.com', ()) in tstore.certs
        assert ('two.com', ()) in tstore.certs
        assert ('three.com', ()) in tstore.certs
        tstore.get_cert('four.com', [])
        assert ('one.com', ()) not in tstore.certs
        assert ('two.com', ()) in tstore.certs
        assert ('three.com', ()) in tstore.certs
        assert ('four.com', ()) in tstore.certs

    def test_overrides(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        ca1 = certs.CertStore.from_store(tmp_path / 'ca1', 'test', 2048)
        ca2 = certs.CertStore.from_store(tmp_path / 'ca2', 'test', 2048)
        assert not ca1.default_ca.serial == ca2.default_ca.serial
        dc = ca2.get_cert('foo.com', ['sans.example.com'])
        dcp = tmp_path / 'dc'
        dcp.write_bytes(dc.cert.to_pem())
        ca1.add_cert_file('foo.com', dcp)
        ret = ca1.get_cert('foo.com', [])
        assert ret.cert.serial == dc.cert.serial

    def test_create_dhparams(self, tmp_path):
        if False:
            print('Hello World!')
        filename = tmp_path / 'dhparam.pem'
        certs.CertStore.load_dhparam(filename)
        assert filename.exists()

    @skip_windows
    def test_umask_secret(self, tmpdir):
        if False:
            i = 10
            return i + 15
        filename = str(tmpdir.join('secret'))
        with certs.CertStore.umask_secret(), open(filename, 'wb'):
            pass
        assert os.stat(filename).st_mode & 63 == 0

class TestDummyCert:

    def test_with_ca(self, tstore):
        if False:
            for i in range(10):
                print('nop')
        r = certs.dummy_cert(tstore.default_privatekey, tstore.default_ca._cert, 'foo.com', ['one.com', 'two.com', '*.three.com', '127.0.0.1', 'bücher.example'], 'Foo Ltd.')
        assert r.cn == 'foo.com'
        assert r.altnames == ['one.com', 'two.com', '*.three.com', 'xn--bcher-kva.example', '127.0.0.1']
        assert r.organization == 'Foo Ltd.'
        r = certs.dummy_cert(tstore.default_privatekey, tstore.default_ca._cert, None, [], None)
        assert r.cn is None
        assert r.organization is None
        assert r.altnames == []

class TestCert:

    def test_simple(self, tdata):
        if False:
            for i in range(10):
                print('nop')
        with open(tdata.path('mitmproxy/net/data/text_cert'), 'rb') as f:
            d = f.read()
        c1 = certs.Cert.from_pem(d)
        assert c1.cn == 'google.com'
        assert len(c1.altnames) == 436
        assert c1.organization == 'Google Inc'
        assert hash(c1)
        with open(tdata.path('mitmproxy/net/data/text_cert_2'), 'rb') as f:
            d = f.read()
        c2 = certs.Cert.from_pem(d)
        assert c2.cn == 'www.inode.co.nz'
        assert len(c2.altnames) == 2
        assert c2.fingerprint()
        assert c2.notbefore == datetime(year=2010, month=1, day=11, hour=19, minute=27, second=36, tzinfo=timezone.utc)
        assert c2.notafter == datetime(year=2011, month=1, day=12, hour=9, minute=14, second=55, tzinfo=timezone.utc)
        assert c2.subject
        assert c2.keyinfo == ('RSA', 2048)
        assert c2.serial
        assert c2.issuer
        assert c2.to_pem()
        assert c2.has_expired() is not None
        assert repr(c2) == "<Cert(cn='www.inode.co.nz', altnames=['www.inode.co.nz', 'inode.co.nz'])>"
        assert c1 != c2

    def test_convert(self, tdata):
        if False:
            for i in range(10):
                print('nop')
        with open(tdata.path('mitmproxy/net/data/text_cert'), 'rb') as f:
            d = f.read()
        c = certs.Cert.from_pem(d)
        assert c == certs.Cert.from_pem(c.to_pem())
        assert c == certs.Cert.from_state(c.get_state())
        assert c == certs.Cert.from_pyopenssl(c.to_pyopenssl())

    @pytest.mark.parametrize('filename,name,bits', [('text_cert', 'RSA', 1024), ('dsa_cert.pem', 'DSA', 1024), ('ec_cert.pem', 'EC (secp256r1)', 256)])
    def test_keyinfo(self, tdata, filename, name, bits):
        if False:
            return 10
        with open(tdata.path(f'mitmproxy/net/data/{filename}'), 'rb') as f:
            d = f.read()
        c = certs.Cert.from_pem(d)
        assert c.keyinfo == (name, bits)

    def test_err_broken_sans(self, tdata):
        if False:
            while True:
                i = 10
        with open(tdata.path('mitmproxy/net/data/text_cert_weird1'), 'rb') as f:
            d = f.read()
        c = certs.Cert.from_pem(d)
        assert c.altnames is not None

    def test_state(self, tdata):
        if False:
            return 10
        with open(tdata.path('mitmproxy/net/data/text_cert'), 'rb') as f:
            d = f.read()
        c = certs.Cert.from_pem(d)
        c.get_state()
        c2 = c.copy()
        a = c.get_state()
        b = c2.get_state()
        assert a == b
        assert c == c2
        assert c is not c2
        c2.set_state(a)
        assert c == c2

    def test_from_store_with_passphrase(self, tdata, tstore):
        if False:
            while True:
                i = 10
        tstore.add_cert_file('unencrypted-no-pass', Path(tdata.path('mitmproxy/data/testkey.pem')), None)
        tstore.add_cert_file('unencrypted-pass', Path(tdata.path('mitmproxy/data/testkey.pem')), b'password')
        tstore.add_cert_file('encrypted-pass', Path(tdata.path('mitmproxy/data/mitmproxy.pem')), b'password')
        with pytest.raises(TypeError):
            tstore.add_cert_file('encrypted-no-pass', Path(tdata.path('mitmproxy/data/mitmproxy.pem')), None)

    def test_special_character(self, tdata):
        if False:
            i = 10
            return i + 15
        with open(tdata.path('mitmproxy/net/data/text_cert_with_comma'), 'rb') as f:
            d = f.read()
        c = certs.Cert.from_pem(d)
        assert dict(c.issuer).get('O') == 'DigiCert, Inc.'
        assert dict(c.subject).get('O') == 'GitHub, Inc.'

    def test_multi_valued_rdns(self, tdata):
        if False:
            for i in range(10):
                print('nop')
        subject = x509.Name([x509.RelativeDistinguishedName([x509.NameAttribute(NameOID.TITLE, 'Test'), x509.NameAttribute(NameOID.COMMON_NAME, 'Multivalue'), x509.NameAttribute(NameOID.SURNAME, 'RDNs'), x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'TSLA')]), x509.RelativeDistinguishedName([x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'PyCA')])])
        expected = [('2.5.4.12', 'Test'), ('CN', 'Multivalue'), ('2.5.4.4', 'RDNs'), ('O', 'TSLA'), ('O', 'PyCA')]
        assert certs._name_to_keyval(subject) == expected