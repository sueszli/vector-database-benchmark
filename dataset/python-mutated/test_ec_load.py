from ..hazmat.primitives.fixtures_ec import EC_KEY_SECP256R1

def test_load_ec_public_numbers(benchmark):
    if False:
        print('Hello World!')
    benchmark(EC_KEY_SECP256R1.public_numbers.public_key)

def test_load_ec_private_numbers(benchmark):
    if False:
        i = 10
        return i + 15
    benchmark(EC_KEY_SECP256R1.private_key)