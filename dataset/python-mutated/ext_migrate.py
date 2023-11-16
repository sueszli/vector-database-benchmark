import flask_migrate

def init(app, db):
    if False:
        print('Hello World!')
    flask_migrate.Migrate(app, db)