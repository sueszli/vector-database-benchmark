import flask_login
login_manager = flask_login.LoginManager()

def init_app(app):
    if False:
        i = 10
        return i + 15
    login_manager.init_app(app)