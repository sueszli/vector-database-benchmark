import bcrypt
from app import User

def handle_signup(username, password):
    if False:
        while True:
            i = 10
    return User.create(username, bcrypt.hashpw(password, bcrypt.getsalt()))

def get_user(user_id):
    if False:
        return 10
    return User.query.filter_by(user_id=user_id).first()