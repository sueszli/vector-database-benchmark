def validate_validity_type(validity_type):
    if False:
        while True:
            i = 10
    '\n    Certificate Validity Type validation rule.\n    Property: Validity.Type\n    '
    VALID_VALIDITY_TYPE = ('ABSOLUTE', 'DAYS', 'END_DATE', 'MONTHS', 'YEARS')
    if validity_type not in VALID_VALIDITY_TYPE:
        raise ValueError('Certificate Validity Type must be one of: %s' % ', '.join(VALID_VALIDITY_TYPE))
    return validity_type

def validate_signing_algorithm(signing_algorithm):
    if False:
        for i in range(10):
            print('nop')
    '\n    Certificate SigningAlgorithm validation rule.\n    Property: Certificate.SigningAlgorithm\n    Property: CertificateAuthority.SigningAlgorithm\n    '
    VALID_SIGNIN_ALGORITHM = ['SHA256WITHECDSA', 'SHA256WITHRSA', 'SHA384WITHECDSA', 'SHA384WITHRSA', 'SHA512WITHECDSA', 'SHA512WITHRSA']
    if signing_algorithm not in VALID_SIGNIN_ALGORITHM:
        raise ValueError('Certificate SigningAlgorithm must be one of: %s' % ', '.join(VALID_SIGNIN_ALGORITHM))
    return signing_algorithm

def validate_key_algorithm(key_algorithm):
    if False:
        i = 10
        return i + 15
    '\n    CertificateAuthority KeyAlgorithm validation rule.\n    Property: CertificateAuthority.KeyAlgorithm\n    '
    VALID_KEY_ALGORITHM = ('EC_prime256v1', 'EC_secp384r1', 'RSA_2048', 'RSA_4096')
    if key_algorithm not in VALID_KEY_ALGORITHM:
        raise ValueError('CertificateAuthority KeyAlgorithm must be one of: %s' % ', '.join(VALID_KEY_ALGORITHM))
    return key_algorithm

def validate_certificateauthority_type(certificateauthority_type):
    if False:
        for i in range(10):
            print('nop')
    '\n    CertificateAuthority Type validation rule.\n    Property: CertificateAuthority.Type\n    '
    VALID_CERTIFICATEAUTHORITY_TYPE = ('ROOT', 'SUBORDINATE')
    if certificateauthority_type not in VALID_CERTIFICATEAUTHORITY_TYPE:
        raise ValueError('CertificateAuthority Type must be one of: %s' % ', '.join(VALID_CERTIFICATEAUTHORITY_TYPE))
    return certificateauthority_type