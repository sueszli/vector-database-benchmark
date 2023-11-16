from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()

def init_app(app):
    if False:
        print('Hello World!')
    db.init_app(app)