from cryptography.hazmat.primitives.asymmetric import rsa

def rsa_priv(foo, key_size='4096'):
    if False:
        for i in range(10):
            print('nop')
    size = int(key_size)
    key = rsa.generate_private_key(public_exponent=65537, key_size=size, backend=default_backend())