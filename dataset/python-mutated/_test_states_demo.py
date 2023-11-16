import os
import sys
import time
import cherrypy
starttime = time.time()

class Root:

    @cherrypy.expose
    def index(self):
        if False:
            i = 10
            return i + 15
        return 'Hello World'

    @cherrypy.expose
    def mtimes(self):
        if False:
            return 10
        return repr(cherrypy.engine.publish('Autoreloader', 'mtimes'))

    @cherrypy.expose
    def pid(self):
        if False:
            print('Hello World!')
        return str(os.getpid())

    @cherrypy.expose
    def start(self):
        if False:
            i = 10
            return i + 15
        return repr(starttime)

    @cherrypy.expose
    def exit(self):
        if False:
            i = 10
            return i + 15
        cherrypy.engine.wait(state=cherrypy.engine.states.STARTED)
        cherrypy.engine.exit()

@cherrypy.engine.subscribe('start', priority=100)
def unsub_sig():
    if False:
        while True:
            i = 10
    cherrypy.log('unsubsig: %s' % cherrypy.config.get('unsubsig', False))
    if cherrypy.config.get('unsubsig', False):
        cherrypy.log('Unsubscribing the default cherrypy signal handler')
        cherrypy.engine.signal_handler.unsubscribe()
    try:
        from signal import signal, SIGTERM
    except ImportError:
        pass
    else:

        def old_term_handler(signum=None, frame=None):
            if False:
                while True:
                    i = 10
            cherrypy.log('I am an old SIGTERM handler.')
            sys.exit(0)
        cherrypy.log('Subscribing the new one.')
        signal(SIGTERM, old_term_handler)

@cherrypy.engine.subscribe('start', priority=6)
def starterror():
    if False:
        while True:
            i = 10
    if cherrypy.config.get('starterror', False):
        1 / 0

@cherrypy.engine.subscribe('start', priority=6)
def log_test_case_name():
    if False:
        while True:
            i = 10
    if cherrypy.config.get('test_case_name', False):
        cherrypy.log('STARTED FROM: %s' % cherrypy.config.get('test_case_name'))
cherrypy.tree.mount(Root(), '/', {'/': {}})