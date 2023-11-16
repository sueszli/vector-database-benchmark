import base64
from extensions.ext_database import db
from libs import rsa
from models.account import Tenant

def obfuscated_token(token: str):
    if False:
        i = 10
        return i + 15
    return token[:6] + '*' * (len(token) - 8) + token[-2:]

def encrypt_token(tenant_id: str, token: str):
    if False:
        print('Hello World!')
    tenant = db.session.query(Tenant).filter(Tenant.id == tenant_id).first()
    encrypted_token = rsa.encrypt(token, tenant.encrypt_public_key)
    return base64.b64encode(encrypted_token).decode()

def decrypt_token(tenant_id: str, token: str):
    if False:
        i = 10
        return i + 15
    return rsa.decrypt(base64.b64decode(token), tenant_id)