import keyring

def populate_keyring() -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    This method populates the keyring with the credentials for unsplash\n    :return:    None\n    '
    keyring.set_password('unsplash', 'access_key', 'FgpGELvBVuBuz3caU8wLz-_gkKa08hwDMb9QGR5AiMg')