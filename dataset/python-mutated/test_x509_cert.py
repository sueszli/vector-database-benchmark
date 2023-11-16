import os
import unittest
import json
import jc.parsers.x509_cert
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-ca-cert.der'), 'rb') as f:
        x509_ca_cert = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-cert-and-key.pem'), 'r', encoding='utf-8') as f:
        x509_cert_and_key_pem = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-letsencrypt.pem'), 'r', encoding='utf-8') as f:
        x509_letsencrypt = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-multi-cert.pem'), 'r', encoding='utf-8') as f:
        x509_multi_cert = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-string-serialnumber.der'), 'rb') as f:
        x509_string_serialnumber = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-cert-bad-email.pem'), 'rb') as f:
        x509_cert_bad_email = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-cert-superfluous-bits.pem'), 'rb') as f:
        x509_cert_superfluous_bits = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-negative-serial.pem'), 'rb') as f:
        x509_cert_negative_serial = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-ca-cert.json'), 'r', encoding='utf-8') as f:
        x509_ca_cert_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-cert-and-key.json'), 'r', encoding='utf-8') as f:
        x509_cert_and_key_pem_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-letsencrypt.json'), 'r', encoding='utf-8') as f:
        x509_letsencrypt_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-multi-cert.json'), 'r', encoding='utf-8') as f:
        x509_multi_cert_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-string-serialnumber.json'), 'r', encoding='utf-8') as f:
        x509_string_serialnumber_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-cert-bad-email.json'), 'r', encoding='utf-8') as f:
        x509_cert_bad_email_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-cert-superfluous-bits.json'), 'r', encoding='utf-8') as f:
        x509_cert_superfluous_bits_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/x509-negative-serial.json'), 'r', encoding='utf-8') as f:
        x509_cert_negative_serial_json = json.loads(f.read())

    def test_x509_cert_nodata(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'x509_cert' with no data\n        "
        self.assertEqual(jc.parsers.x509_cert.parse('', quiet=True), [])

    def test_x509_ca_cert(self):
        if False:
            return 10
        "\n        Test 'cat x509-ca-cert.der' (CA cert in DER format)\n        "
        self.assertEqual(jc.parsers.x509_cert.parse(self.x509_ca_cert, quiet=True), self.x509_ca_cert_json)

    def test_x509_cert_and_key(self):
        if False:
            return 10
        "\n        Test 'cat x509-cert-and-key.pem' (combo cert and key file in PEM format)\n        "
        self.assertEqual(jc.parsers.x509_cert.parse(self.x509_cert_and_key_pem, quiet=True), self.x509_cert_and_key_pem_json)

    def test_x509_letsencrypt(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'cat x509-letsencrypt.pem' (letsencrypt cert in PEM format)\n        "
        self.assertEqual(jc.parsers.x509_cert.parse(self.x509_letsencrypt, quiet=True), self.x509_letsencrypt_json)

    def test_x509_multi_cert(self):
        if False:
            print('Hello World!')
        "\n        Test 'cat x509-multi-cert.pem' (PEM file with multiple certificates)\n        "
        self.assertEqual(jc.parsers.x509_cert.parse(self.x509_multi_cert, quiet=True), self.x509_multi_cert_json)

    def test_x509_string_serialnumber(self):
        if False:
            while True:
                i = 10
        "\n        Test 'cat x509-string-serialnumber.der' (DER file with string serial numbers)\n        "
        self.assertEqual(jc.parsers.x509_cert.parse(self.x509_multi_cert, quiet=True), self.x509_multi_cert_json)

    def test_x509_cert_bad_email(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'cat x509-cert-bad-email.pem' (PEM file with a non-compliant email address)\n        "
        self.assertEqual(jc.parsers.x509_cert.parse(self.x509_cert_bad_email, quiet=True), self.x509_cert_bad_email_json)

    def test_x509_cert_superfluous_bits(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'cat x509-cert-superfluous-bits.pem' (PEM file with more bits set for the keyUsage extension than defined by the RFC)\n        "
        self.assertEqual(jc.parsers.x509_cert.parse(self.x509_cert_superfluous_bits, quiet=True), self.x509_cert_superfluous_bits_json)

    def test_x509_cert_negative_serial(self):
        if False:
            print('Hello World!')
        "\n        Test 'cat x509-cert-bad-email.pem' (PEM file with a non-compliant email address)\n        "
        self.assertEqual(jc.parsers.x509_cert.parse(self.x509_cert_negative_serial, quiet=True), self.x509_cert_negative_serial_json)
if __name__ == '__main__':
    unittest.main()