import sys
import logging
import inspect
import traceback
from core.logger import logger
formatter = logging.Formatter('%(asctime)s [ProxyPlugins] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logger().setup_logger('ProxyPlugins', formatter)

class ProxyPlugins:
    """
    This class does some magic so that all we need to do in
    ServerConnection is do a self.plugins.hook() call
    and we will call any plugin that implements the function
    that it came from with the args passed to the original
    function.

    To do this, we are probably abusing the inspect module,
    and if it turns out to be too slow it can be changed. For
    now, it's nice because it makes for very little code needed
    to tie us in.

    Sadly, propagating changes back to the function is not quite
    as easy in all cases :-/ . Right now, changes to local function
    vars still have to be set back in the function. This only happens
    in handleResponse, but is still annoying.
    """
    mthdDict = {'connectionMade': 'request', 'handleStatus': 'responsestatus', 'handleResponse': 'response', 'handleHeader': 'responseheaders', 'handleEndHeaders': 'responseheaders'}
    plugin_mthds = {}
    plugin_list = []
    all_plugins = []
    __shared_state = {}

    def __init__(self):
        if False:
            return 10
        self.__dict__ = self.__shared_state

    def set_plugins(self, plugins):
        if False:
            return 10
        'Set the plugins in use'
        for p in plugins:
            self.add_plugin(p)
        log.debug('Loaded {} plugin/s'.format(len(plugins)))

    def add_plugin(self, p):
        if False:
            return 10
        'Load a plugin'
        self.plugin_list.append(p)
        log.debug('Adding {} plugin'.format(p.name))
        for (mthd, pmthd) in self.mthdDict.iteritems():
            try:
                self.plugin_mthds[mthd].append(getattr(p, pmthd))
            except KeyError:
                self.plugin_mthds[mthd] = [getattr(p, pmthd)]

    def remove_plugin(self, p):
        if False:
            i = 10
            return i + 15
        'Unload a plugin'
        self.plugin_list.remove(p)
        log.debug('Removing {} plugin'.format(p.name))
        for (mthd, pmthd) in self.mthdDict.iteritems():
            try:
                self.plugin_mthds[mthd].remove(getattr(p, pmthd))
            except KeyError:
                pass

    def hook(self):
        if False:
            while True:
                i = 10
        'Magic to hook various function calls in sslstrip'
        frame = sys._getframe(1)
        fname = frame.f_code.co_name
        (keys, _, _, values) = inspect.getargvalues(frame)
        args = {}
        for key in keys:
            args[key] = values[key]
        if fname == 'handleResponse' or fname == 'handleHeader' or fname == 'handleEndHeaders':
            args['request'] = args['self']
            args['response'] = args['self'].client
        else:
            args['request'] = args['self']
        del args['self']
        log.debug('hooking {}()'.format(fname))
        try:
            if self.plugin_mthds:
                for f in self.plugin_mthds[fname]:
                    a = f(**args)
                    if a != None:
                        args = a
        except Exception as e:
            log.error('Exception occurred in hooked function')
            traceback.print_exc()
        return args