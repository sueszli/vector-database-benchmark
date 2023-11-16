import re
import time
from datetime import timedelta
from pyload.core.utils.purge import chars as remove_chars
from pyload.core.utils.purge import uniquify
from .account import BaseAccount

class MultiAccount(BaseAccount):
    __name__ = 'MultiAccount'
    __type__ = 'account'
    __version__ = '0.24'
    __status__ = 'testing'
    __config__ = [('enabled', 'bool', 'Activated', True), ('mh_mode', 'all;listed;unlisted', 'Filter downloaders to use', 'all'), ('mh_list', 'str', 'Downloader list (comma separated)', ''), ('mh_interval', 'int', 'Reload interval in hours', 12)]
    __description__ = 'Multi-downloader account plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com'), ('GammaC0de', 'nitzo2001[AT]yahoo[DOT]com')]
    DOMAIN_REPLACEMENTS = [('ddl\\.to', 'ddownload.com'), ('180upload\\.com', 'hundredeightyupload.com'), ('bayfiles\\.net', 'bayfiles.com'), ('cloudnator\\.com', 'shragle.com'), ('dfiles\\.eu', 'depositfiles.com'), ('easy-share\\.com', 'crocko.com'), ('freakshare\\.net', 'freakshare.com'), ('hellshare\\.com', 'hellshare.cz'), ('ifile\\.it', 'filecloud.io'), ('nowdownload\\.\\w+', 'nowdownload.sx'), ('nowvideo\\.\\w+', 'nowvideo.sx'), ('putlocker\\.com', 'firedrive.com'), ('share-?rapid\\.cz', 'multishare.cz'), ('ul\\.to', 'uploaded.to'), ('uploaded\\.net', 'uploaded.to'), ('uploadhero\\.co', 'uploadhero.com'), ('zshares\\.net', 'zshare.net'), ('^1', 'one'), ('^2', 'two'), ('^3', 'three'), ('^4', 'four'), ('^5', 'five'), ('^6', 'six'), ('^7', 'seven'), ('^8', 'eight'), ('^9', 'nine'), ('^0', 'zero')]

    def init(self):
        if False:
            print('Hello World!')
        self.need_reactivate = False
        self.plugins = []
        self.supported = []
        self.pluginclass = None
        self.pluginmodule = None
        self.plugintype = None
        self.fail_count = 0
        self.init_plugin()

    def init_plugin(self):
        if False:
            print('Hello World!')
        (plugin, self.plugintype) = self.pyload.plugin_manager.find_plugin(self.classname)
        if plugin:
            self.pluginmodule = self.pyload.plugin_manager.load_module(self.plugintype, self.classname)
            self.pluginclass = self.pyload.plugin_manager.load_class(self.plugintype, self.classname)
            self.pyload.addon_manager.add_event('plugin_updated', self.plugins_updated)
            self.periodical.start(3, threaded=True)
        else:
            self.log_warning(self._('Multi-downloader feature will be deactivated due missing plugin reference'))

    def plugins_updated(self, type_plugins):
        if False:
            return 10
        if not any((t in ('base', 'addon') for (t, n) in type_plugins)):
            self.reactivate()

    def periodical_task(self):
        if False:
            for i in range(10):
                print('nop')
        self.reactivate(refresh=True)

    def replace_domains(self, list):
        if False:
            return 10
        for r in self.DOMAIN_REPLACEMENTS:
            (pattern, repl) = r
            _re = re.compile(pattern, re.I | re.U)
            list = [_re.sub(repl, domain) if _re.match(domain) else domain for domain in list]
        return list

    def parse_domains(self, list):
        if False:
            return 10
        _re = re.compile('^(?:https?://)?(?:www\\.)?(?:\\w+\\.)*((?:(?:\\d{1,3}\\.){3}\\d{1,3}|[\\w\\-^_]{3,63}(?:\\.[a-zA-Z]{2,}){1,2})(?:\\:\\d+)?)', re.I | re.U)
        domains = [domain.strip().lower() for url in list for domain in _re.findall(url)]
        return self.replace_domains(uniquify(domains))

    def _grab_hosters(self):
        if False:
            print('Hello World!')
        self.info['data']['hosters'] = []
        try:
            hosterlist = self.grab_hosters(self.user, self.info['login']['password'], self.info['data'])
            if hosterlist and isinstance(hosterlist, list):
                domains = self.parse_domains(hosterlist)
                self.info['data']['hosters'] = sorted(domains)
                self.sync(reverse=True)
        except Exception as exc:
            self.log_warning(self._('Error loading downloader list for user `{}`').format(self.user), exc, exc_info=self.pyload.debug > 1, stack_info=self.pyload.debug > 2)
        finally:
            self.log_debug('Downloader list for user `{}`: {}'.format(self.user, self.info['data']['hosters']))
            return self.info['data']['hosters']

    def grab_hosters(self, user, password, data):
        if False:
            while True:
                i = 10
        '\n        Load list of supported downloaders.\n\n        :return: List of domain names\n        '
        raise NotImplementedError

    def _override(self):
        if False:
            return 10
        prev_supported = self.supported
        new_supported = []
        excluded = []
        self.supported = []
        if self.plugintype == 'downloader':
            plugin_map = {name.lower(): name for name in self.pyload.plugin_manager.downloader_plugins.keys()}
            account_list = [account.type.lower() for account in self.pyload.api.get_accounts(False) if account.valid and account.premium]
        else:
            plugin_map = {}
            account_list = [name[::-1].replace('Folder'[::-1], '', 1).lower()[::-1] for name in self.pyload.plugin_manager.decrypter_plugins.keys()]
        for plugin in self.get_plugins():
            name = remove_chars(plugin, '-.')
            if name in account_list:
                excluded.append(plugin)
            elif name in plugin_map:
                self.supported.append(plugin_map[name])
            else:
                new_supported.append(plugin)
        removed = [plugin for plugin in prev_supported if plugin not in self.supported]
        if removed:
            self.log_debug(f"Unload: {', '.join(removed)}")
            for plugin in removed:
                self.unload_plugin(plugin)
        if not self.supported and (not new_supported):
            self.log_error(self._('No {} loaded').format(self.plugintype))
            return
        self.log_debug('Overwritten {}s: {}'.format(self.plugintype, ', '.join(sorted(self.supported))))
        for plugin in self.supported:
            hdict = self.pyload.plugin_manager.plugins[self.plugintype][plugin]
            hdict['new_module'] = self.pluginmodule
            hdict['new_name'] = self.classname
        if excluded:
            self.log_info(self._('{}s not overwritten: {}').format(self.plugintype.capitalize(), ', '.join(sorted(excluded))))
        if new_supported:
            plugins = sorted(new_supported)
            self.log_debug(f"New {self.plugintype}s: {', '.join(plugins)}")
            domains = '|'.join((x.replace('.', '\\.') for x in plugins))
            pattern = f'.*(?P<DOMAIN>{domains}).*'
            if hasattr(self.pluginclass, '__pattern__') and isinstance(self.pluginclass.__pattern__, str) and ('://' in self.pluginclass.__pattern__):
                pattern = f'{self.pluginclass.__pattern__}|{pattern}'
            self.log_debug(f'Pattern: {pattern}')
            hdict = self.pyload.plugin_manager.plugins[self.plugintype][self.classname]
            hdict['pattern'] = pattern
            hdict['re'] = re.compile(pattern)

    def get_plugins(self, cached=True):
        if False:
            while True:
                i = 10
        if cached and self.plugins:
            return self.plugins
        for _ in range(5):
            try:
                plugin_set = set(self._grab_hosters())
                break
            except Exception as exc:
                self.log_warning(exc, self._('Waiting 1 minute and retry'), exc_info=self.pyload.debug > 1, stack_info=self.pyload.debug > 2)
                time.sleep(60)
        else:
            self.log_warning(self._('No hoster list retrieved'))
            return []
        try:
            mh_mode = self.config.get('mh_mode', 'all')
            if mh_mode in ('listed', 'unlisted'):
                mh_list = self.config.get('mh_list', '').replace('|', ',').replace(';', ',').split(',')
                config_set = set(mh_list)
                if mh_mode == 'listed':
                    plugin_set &= config_set
                else:
                    plugin_set -= config_set
        except Exception as exc:
            self.log_error(exc)
        self.plugins = list(plugin_set)
        return self.plugins

    def unload_plugin(self, plugin):
        if False:
            print('Hello World!')
        hdict = self.pyload.plugin_manager.plugins[self.plugintype][plugin]
        if 'pyload' in hdict:
            hdict.pop('pyload', None)
        if 'new_module' in hdict:
            hdict.pop('new_module', None)
            hdict.pop('new_name', None)

    def reactivate(self, refresh=False):
        if False:
            i = 10
            return i + 15
        reloading = self.info['data'].get('hosters') is not None
        if self.info['login']['valid'] is None:
            return
        else:
            interval = self.config.get('mh_interval', 12) * 60 * 60
            self.periodical.set_interval(interval)
        if self.info['login']['valid'] is False:
            self.fail_count += 1
            if self.fail_count < 3:
                if reloading:
                    self.log_error(self._('Could not reload hoster list - invalid account, retry in 5 minutes'))
                else:
                    self.log_error(self._('Could not load hoster list - invalid account, retry in 5 minutes'))
                self.periodical.set_interval(timedelta(minutes=5).total_seconds())
            else:
                if reloading:
                    self.log_error(self._('Could not reload hoster list - invalid account, deactivating'))
                else:
                    self.log_error(self._('Could not load hoster list - invalid account, deactivating'))
                self.deactivate()
            return
        if not self.logged:
            if not self.relogin():
                self.fail_count += 1
                if self.fail_count < 3:
                    if reloading:
                        self.log_error(self._('Could not reload hoster list - login failed, retry in 5 minutes'))
                    else:
                        self.log_error(self._('Could not load hoster list - login failed, retry in 5 minutes'))
                    self.periodical.set_interval(timedelta(minutes=5).total_seconds())
                else:
                    if reloading:
                        self.log_error(self._('Could not reload hoster list - login failed, deactivating'))
                    else:
                        self.log_error(self._('Could not load hoster list - login failed, deactivating'))
                    self.deactivate()
                return
        self.pyload.addon_manager.add_event('plugin_updated', self.plugins_updated)
        if refresh or not reloading:
            if not self.get_plugins(cached=False):
                self.fail_count += 1
                if self.fail_count < 3:
                    self.log_error(self._('Failed to load hoster list for user `{}`, retry in 5 minutes').format(self.user))
                    self.periodical.set_interval(timedelta(minutes=5).total_seconds())
                else:
                    self.log_error(self._('Failed to load hoster list for user `{}`, deactivating').format(self.user))
                    self.deactivate()
                return
        if self.fail_count:
            self.fail_count = 0
            interval = timedelta(hours=self.config.get('mh_interval', 12)).total_seconds()
            self.periodical.set_interval(interval)
        self._override()

    def deactivate(self):
        if False:
            print('Hello World!')
        '\n        Remove override for all plugins.\n        '
        self.log_info(self._('Reverting back to default hosters'))
        self.pyload.addon_manager.remove_event('plugin_updated', self.plugins_updated)
        self.periodical.stop()
        self.fail_count = 0
        if self.supported:
            self.log_debug(f"Unload: {', '.join(self.supported)}")
            for plugin in self.supported:
                self.unload_plugin(plugin)
        hdict = self.pyload.plugin_manager.plugins[self.plugintype][self.classname]
        hdict['pattern'] = getattr(self.pluginclass, '__pattern__', '^unmatchable$')
        hdict['re'] = re.compile(hdict['pattern'])

    def update_accounts(self, user, password=None, options={}):
        if False:
            while True:
                i = 10
        super().update_accounts(user, password, options)
        if self.need_reactivate:
            interval = timedelta(hours=self.config.get('mh_interval', 12)).total_seconds()
            self.periodical.restart(interval, threaded=True, delay=2)
        self.need_reactivate = True

    def remove_account(self, user):
        if False:
            while True:
                i = 10
        self.deactivate()
        super().remove_account(user)