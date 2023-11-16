import rsa
(tambe_pub, tambe_priv) = rsa.newkeys(512)
message = 'beautiful people!'.encode('utf8')
encrypted_msg = rsa.encrypt(message, tambe_pub)
message = rsa.decrypt(encrypted_msg, tambe_priv)

def encryption_decryption():
    if False:
        i = 10
        return i + 15
    'Function to test encryption and decryption'
    assert message.decode('utf8') == 'beautiful people!'
encryption_decryption()
print('OK.')