"""This application verifies HSM attestations using certificate bundles
obtained from Cloud HSM.

For more information, visit https://cloud.google.com/kms/docs/attest-key.
"""
import argparse
import gzip
from cryptography import exceptions
from cryptography import x509
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import padding
import pem

def verify(attestation_file, bundle_file):
    if False:
        return 10
    'Verifies an attestation using a bundle of certificates.\n\n    Args:\n      attestation_file: The name of the attestation file.\n      bundle_file: The name of the bundle file containing the certificates\n        used to verify the attestation.\n\n    Returns:\n      True if at least one of the certificates in bundle_file can verify the\n      attestation data and its signature.\n    '
    with gzip.open(attestation_file, 'rb') as f:
        attestation = f.read()
        data = attestation[:-256]
        signature = attestation[-256:]
        for cert in pem.parse_file(bundle_file):
            cert_obj = x509.load_pem_x509_certificate(str(cert).encode('utf-8'), backends.default_backend())
            try:
                cert_obj.public_key().verify(signature, data, padding.PKCS1v15(), cert_obj.signature_hash_algorithm)
                return True
            except exceptions.InvalidSignature:
                continue
        return False
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attestation_file', help='Name of attestation file.')
    parser.add_argument('bundle_file', help='Name of certificate bundle file.')
    args = parser.parse_args()
    if verify(args.attestation_file, args.bundle_file):
        print('Signature verified.')
    else:
        print('Signature verification failed.')