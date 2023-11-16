import ipaddress
import time
import OpenSSL
import mitmproxy
from mitmproxy import ctx
from mitmproxy.certs import Cert

def monkey_dummy_cert(privkey, cacert, commonname, sans):
    if False:
        print('Hello World!')
    ss = []
    for i in sans:
        try:
            ipaddress.ip_address(i.decode('ascii'))
        except ValueError:
            if ctx.options.certwrongCN:
                ss.append(b'DNS:%sm' % i)
            else:
                ss.append(b'DNS:%s' % i)
        else:
            ss.append(b'IP:%s' % i)
    ss = b', '.join(ss)
    cert = OpenSSL.crypto.X509()
    if ctx.options.certbeginon:
        cert.gmtime_adj_notBefore(3600 * 48)
    else:
        cert.gmtime_adj_notBefore(-3600 * 48)
    if ctx.options.certexpire:
        cert.gmtime_adj_notAfter(-3600 * 24)
    else:
        cert.gmtime_adj_notAfter(94608000)
    cert.set_issuer(cacert.get_subject())
    if commonname is not None and len(commonname) < 64:
        if ctx.options.certwrongCN:
            new_cn = commonname + b'm'
            cert.get_subject().CN = new_cn
        else:
            cert.get_subject().CN = commonname
    cert.set_serial_number(int(time.time() * 10000))
    if ss:
        cert.set_version(2)
        cert.add_extensions([OpenSSL.crypto.X509Extension(b'subjectAltName', False, ss)])
        cert.set_pubkey(cacert.get_pubkey())
        cert.sign(privkey, 'sha256')
        return Cert(cert)

class CheckSSLPinning:

    def load(self, loader):
        if False:
            for i in range(10):
                print('nop')
        loader.add_option('certbeginon', bool, False, "\n            Sets SSL Certificate's 'Begins On' time in future.\n            ")
        loader.add_option('certexpire', bool, False, "\n            Sets SSL Certificate's 'Expires On' time in the past.\n            ")
        loader.add_option('certwrongCN', bool, False, "\n            Sets SSL Certificate's CommonName(CN) different from the domain name.\n            ")

    def clientconnect(self, layer):
        if False:
            return 10
        mitmproxy.certs.dummy_cert = monkey_dummy_cert