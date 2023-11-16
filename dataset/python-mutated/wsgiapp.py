import os
from gunicorn.errors import ConfigError
from gunicorn.app.base import Application
from gunicorn import util

class WSGIApplication(Application):

    def init(self, parser, opts, args):
        if False:
            while True:
                i = 10
        self.app_uri = None
        if opts.paste:
            from .pasterapp import has_logging_config
            config_uri = os.path.abspath(opts.paste)
            config_file = config_uri.split('#')[0]
            if not os.path.exists(config_file):
                raise ConfigError('%r not found' % config_file)
            self.cfg.set('default_proc_name', config_file)
            self.app_uri = config_uri
            if has_logging_config(config_file):
                self.cfg.set('logconfig', config_file)
            return
        if len(args) > 0:
            self.cfg.set('default_proc_name', args[0])
            self.app_uri = args[0]

    def load_config(self):
        if False:
            for i in range(10):
                print('nop')
        super().load_config()
        if self.app_uri is None:
            if self.cfg.wsgi_app is not None:
                self.app_uri = self.cfg.wsgi_app
            else:
                raise ConfigError('No application module specified.')

    def load_wsgiapp(self):
        if False:
            i = 10
            return i + 15
        return util.import_app(self.app_uri)

    def load_pasteapp(self):
        if False:
            while True:
                i = 10
        from .pasterapp import get_wsgi_app
        return get_wsgi_app(self.app_uri, defaults=self.cfg.paste_global_conf)

    def load(self):
        if False:
            while True:
                i = 10
        if self.cfg.paste is not None:
            return self.load_pasteapp()
        else:
            return self.load_wsgiapp()

def run():
    if False:
        for i in range(10):
            print('nop')
    '    The ``gunicorn`` command line runner for launching Gunicorn with\n    generic WSGI applications.\n    '
    from gunicorn.app.wsgiapp import WSGIApplication
    WSGIApplication('%(prog)s [OPTIONS] [APP_MODULE]').run()
if __name__ == '__main__':
    run()