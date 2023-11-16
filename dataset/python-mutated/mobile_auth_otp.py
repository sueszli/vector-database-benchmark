from zerver.models import UserProfile

def xor_hex_strings(bytes_a: str, bytes_b: str) -> str:
    if False:
        while True:
            i = 10
    'Given two hex strings of equal length, return a hex string with\n    the bitwise xor of the two hex strings.'
    assert len(bytes_a) == len(bytes_b)
    return ''.join((f'{int(x, 16) ^ int(y, 16):x}' for (x, y) in zip(bytes_a, bytes_b)))

def ascii_to_hex(input_string: str) -> str:
    if False:
        print('Hello World!')
    'Given an ascii string, encode it as a hex string'
    return input_string.encode().hex()

def hex_to_ascii(input_string: str) -> str:
    if False:
        i = 10
        return i + 15
    'Given a hex array, decode it back to a string'
    return bytes.fromhex(input_string).decode()

def otp_encrypt_api_key(api_key: str, otp: str) -> str:
    if False:
        while True:
            i = 10
    assert len(otp) == UserProfile.API_KEY_LENGTH * 2
    hex_encoded_api_key = ascii_to_hex(api_key)
    assert len(hex_encoded_api_key) == UserProfile.API_KEY_LENGTH * 2
    return xor_hex_strings(hex_encoded_api_key, otp)

def otp_decrypt_api_key(otp_encrypted_api_key: str, otp: str) -> str:
    if False:
        i = 10
        return i + 15
    assert len(otp) == UserProfile.API_KEY_LENGTH * 2
    assert len(otp_encrypted_api_key) == UserProfile.API_KEY_LENGTH * 2
    hex_encoded_api_key = xor_hex_strings(otp_encrypted_api_key, otp)
    return hex_to_ascii(hex_encoded_api_key)

def is_valid_otp(otp: str) -> bool:
    if False:
        return 10
    try:
        assert len(otp) == UserProfile.API_KEY_LENGTH * 2
        [int(c, 16) for c in otp]
        return True
    except Exception:
        return False