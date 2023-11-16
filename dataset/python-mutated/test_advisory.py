import pytest
import jwt
from jwt.algorithms import get_default_algorithms
from jwt.exceptions import InvalidKeyError
from .utils import crypto_required
priv_key_bytes = b'-----BEGIN PRIVATE KEY-----\nMC4CAQAwBQYDK2VwBCIEIIbBhdo2ah7X32i50GOzrCr4acZTe6BezUdRIixjTAdL\n-----END PRIVATE KEY-----'
pub_key_bytes = b'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPL1I9oiq+B8crkmuV4YViiUnhdLjCp3hvy1bNGuGfNL'
ssh_priv_key_bytes = b'-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIOWc7RbaNswMtNtc+n6WZDlUblMr2FBPo79fcGXsJlGQoAoGCCqGSM49\nAwEHoUQDQgAElcy2RSSSgn2RA/xCGko79N+7FwoLZr3Z0ij/ENjow2XpUDwwKEKk\nAk3TDXC9U8nipMlGcY7sDpXp2XyhHEM+Rw==\n-----END EC PRIVATE KEY-----'
ssh_key_bytes = b'ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBJXMtkUkkoJ9kQP8QhpKO/TfuxcKC2a92dIo/xDY6MNl6VA8MChCpAJN0w1wvVPJ4qTJRnGO7A6V6dl8oRxDPkc='

class TestAdvisory:

    @crypto_required
    def test_ghsa_ffqj_6fqr_9h24(self):
        if False:
            while True:
                i = 10
        encoded_good = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJFZERTQSJ9.eyJ0ZXN0IjoxMjM0fQ.M5y1EEavZkHSlj9i8yi9nXKKyPBSAUhDRTOYZi3zZY11tZItDaR3qwAye8pc74_lZY3Ogt9KPNFbVOSGnUBHDg'
        encoded_bad = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0ZXN0IjoxMjM0fQ.6ulDpqSlbHmQ8bZXhZRLFko9SwcHrghCwh8d-exJEE4'
        algorithm_names = list(get_default_algorithms())
        jwt.decode(encoded_good, pub_key_bytes, algorithms=algorithm_names)
        with pytest.raises(InvalidKeyError):
            jwt.decode(encoded_bad, pub_key_bytes, algorithms=algorithm_names)
        encoded_good = 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0ZXN0IjoxMjM0fQ.NX42mS8cNqYoL3FOW9ZcKw8Nfq2mb6GqJVADeMA1-kyHAclilYo_edhdM_5eav9tBRQTlL0XMeu_WFE_mz3OXg'
        encoded_bad = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0ZXN0IjoxMjM0fQ.5eYfbrbeGYmWfypQ6rMWXNZ8bdHcqKng5GPr9MJZITU'
        algorithm_names = list(get_default_algorithms())
        jwt.decode(encoded_good, ssh_key_bytes, algorithms=algorithm_names)
        with pytest.raises(InvalidKeyError):
            jwt.decode(encoded_bad, ssh_key_bytes, algorithms=algorithm_names)