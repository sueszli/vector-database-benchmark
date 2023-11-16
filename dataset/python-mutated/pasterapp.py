import configparser
import os
from paste.deploy import loadapp
from gunicorn.app.wsgiapp import WSGIApplication
from gunicorn.config import get_default_config_file

def get_wsgi_app(config_uri, name=None, defaults=None):
    if False:
        while True:
            i = 10
    if ':' not in config_uri:
        config_uri = 'config:%s' % config_uri
    return loadapp(config_uri, name=name, relative_to=os.getcwd(), global_conf=defaults)

def has_logging_config(config_file):
    if False:
        i = 10
        return i + 15
    parser = configparser.ConfigParser()
    parser.read([config_file])
    return parser.has_section('loggers')

def serve(app, global_conf, **local_conf):
    if False:
        print('Hello World!')
    '    A Paste Deployment server runner.\n\n    Example configuration:\n\n        [server:main]\n        use = egg:gunicorn#main\n        host = 127.0.0.1\n        port = 5000\n    '
    config_file = global_conf['__file__']
    gunicorn_config_file = local_conf.pop('config', None)
    host = local_conf.pop('host', '')
    port = local_conf.pop('port', '')
    if host and port:
        local_conf['bind'] = '%s:%s' % (host, port)
    elif host:
        local_conf['bind'] = host.split(',')

    class PasterServerApplication(WSGIApplication):

        def load_config(self):
            if False:
                i = 10
                return i + 15
            self.cfg.set('default_proc_name', config_file)
            if has_logging_config(config_file):
                self.cfg.set('logconfig', config_file)
            if gunicorn_config_file:
                self.load_config_from_file(gunicorn_config_file)
            else:
                default_gunicorn_config_file = get_default_config_file()
                if default_gunicorn_config_file is not None:
                    self.load_config_from_file(default_gunicorn_config_file)
            for (k, v) in local_conf.items():
                if v is not None:
                    self.cfg.set(k.lower(), v)

        def load(self):
            if False:
                print('Hello World!')
            return app
    PasterServerApplication().run()