import sys
from string import digits, ascii_uppercase, ascii_lowercase
from itertools import product
from typing import Callable
from functools import partial
from electrum.wallet import Wallet, Abstract_Wallet
from electrum.storage import WalletStorage
from electrum.wallet_db import WalletDB
from electrum.simple_config import SimpleConfig
from electrum.util import InvalidPassword
ALLOWED_CHARS = digits + ascii_uppercase + ascii_lowercase
MAX_PASSWORD_LEN = 12

def test_password_for_storage_encryption(storage: WalletStorage, password: str) -> bool:
    if False:
        print('Hello World!')
    try:
        storage.decrypt(password)
    except InvalidPassword:
        return False
    else:
        return True

def test_password_for_keystore_encryption(wallet: Abstract_Wallet, password: str) -> bool:
    if False:
        return 10
    try:
        wallet.check_password(password)
    except InvalidPassword:
        return False
    else:
        return True

def bruteforce_loop(test_password: Callable[[str], bool]) -> str:
    if False:
        i = 10
        return i + 15
    num_tested = 0
    for pw_len in range(1, MAX_PASSWORD_LEN + 1):
        for pw_tuple in product(ALLOWED_CHARS, repeat=pw_len):
            password = ''.join(pw_tuple)
            if test_password(password):
                return password
            num_tested += 1
            if num_tested % 5000 == 0:
                print(f'> tested {num_tested} passwords so far... most recently tried: {password!r}')
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('ERROR. usage: bruteforce_pw.py <path_to_wallet_file>')
        sys.exit(1)
    path = sys.argv[1]
    config = SimpleConfig()
    storage = WalletStorage(path)
    if not storage.file_exists():
        print(f'ERROR. wallet file not found at path: {path}')
        sys.exit(1)
    if storage.is_encrypted():
        test_password = partial(test_password_for_storage_encryption, storage)
        print(f'wallet found: with storage encryption.')
    else:
        db = WalletDB(storage.read(), manual_upgrades=True)
        wallet = Wallet(db, storage, config=config)
        if not wallet.has_password():
            print('wallet found but it is not encrypted.')
            sys.exit(0)
        test_password = partial(test_password_for_keystore_encryption, wallet)
        print(f'wallet found: with keystore encryption.')
    password = bruteforce_loop(test_password)
    print(f'====================')
    print(f'password found: {password}')