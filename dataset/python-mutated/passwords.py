import bcrypt

def generate_salt() -> str:
    if False:
        while True:
            i = 10
    return bcrypt.gensalt(14)

def create_bcrypt_hash(password: str, salt: str) -> str:
    if False:
        print('Hello World!')
    password_bytes = password.encode()
    password_hash_bytes = bcrypt.hashpw(password_bytes, salt)
    password_hash_str = password_hash_bytes.decode()
    return password_hash_str

def verify_password(password: str, hash_from_database: str) -> bool:
    if False:
        return 10
    password_bytes = password.encode()
    hash_bytes = hash_from_database.encode()
    does_match = bcrypt.checkpw(password_bytes, hash_bytes)
    return does_match