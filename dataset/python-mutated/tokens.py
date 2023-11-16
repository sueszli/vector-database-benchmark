from flask import jsonify
from app import db
from app.api import bp
from app.api.auth import basic_auth, token_auth

@bp.route('/tokens', methods=['POST'])
@basic_auth.login_required
def get_token():
    if False:
        while True:
            i = 10
    token = basic_auth.current_user().get_token()
    db.session.commit()
    return jsonify({'token': token})

@bp.route('/tokens', methods=['DELETE'])
@token_auth.login_required
def revoke_token():
    if False:
        i = 10
        return i + 15
    token_auth.current_user().revoke_token()
    db.session.commit()
    return ('', 204)