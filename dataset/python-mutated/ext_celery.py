from celery import Task, Celery
from flask import Flask

def init_app(app: Flask) -> Celery:
    if False:
        return 10

    class FlaskTask(Task):

        def __call__(self, *args: object, **kwargs: object) -> object:
            if False:
                print('Hello World!')
            with app.app_context():
                return self.run(*args, **kwargs)
    celery_app = Celery(app.name, task_cls=FlaskTask, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_BACKEND'], task_ignore_result=True)
    ssl_options = {'ssl_cert_reqs': None, 'ssl_ca_certs': None, 'ssl_certfile': None, 'ssl_keyfile': None}
    celery_app.conf.update(result_backend=app.config['CELERY_RESULT_BACKEND'])
    if app.config['BROKER_USE_SSL']:
        celery_app.conf.update(broker_use_ssl=ssl_options)
    celery_app.set_default()
    app.extensions['celery'] = celery_app
    return celery_app