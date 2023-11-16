import time
from datetime import timedelta
from ..api import OnlineStatus
from ..datatypes.pyfile import PyFile
from ..utils.old.packagetools import parse_names
from .plugin_thread import PluginThread

class InfoThread(PluginThread):

    def __init__(self, manager, data, pid=-1, rid=-1, add=False):
        if False:
            i = 10
            return i + 15
        '\n        Constructor.\n        '
        super().__init__(manager)
        self.data = data
        self.pid = pid
        self.rid = rid
        self.add = add
        self.cache = []
        self.start()

    def run(self):
        if False:
            return 10
        '\n        run method.\n        '
        plugins = {}
        container = []
        for (url, plugin) in self.data:
            if plugin in plugins:
                plugins[plugin].append(url)
            else:
                plugins[plugin] = [url]
        for name in self.pyload.plugin_manager.container_plugins:
            if name in plugins:
                container.extend(((name, url) for url in plugins[name]))
                del plugins[name]
        if self.pid > -1:
            for (pluginname, urls) in plugins.items():
                plugin = self.pyload.plugin_manager.get_plugin(pluginname, True)
                if hasattr(plugin, 'get_info'):
                    self.fetch_for_plugin(pluginname, plugin, urls, self.update_db)
                    self.pyload.files.save()
        elif self.add:
            for (pluginname, urls) in plugins.items():
                plugin = self.pyload.plugin_manager.get_plugin(pluginname, True)
                if hasattr(plugin, 'get_info'):
                    self.fetch_for_plugin(pluginname, plugin, urls, self.update_cache, True)
                else:
                    result = [(url, 0, 3, url) for url in urls]
                    self.update_cache(pluginname, result)
            packs = parse_names(((name, url) for (name, x, y, url) in self.cache))
            self.pyload.log.debug(f'Fetched and generated {len(packs)} packages')
            for (k, v) in packs.items():
                self.pyload.api.add_package(k, v)
            del self.cache[:]
        else:
            for (name, url) in container:
                try:
                    data = self.decrypt_container(name, url)
                except Exception:
                    self.pyload.log.warning('Could not decrypt container.', exc_info=self.pyload.debug > 1, stack_info=self.pyload.debug > 2)
                    data = []
                for (url, plugin) in data:
                    if plugin in plugins:
                        plugins[plugin].append(url)
                    else:
                        plugins[plugin] = [url]
            self.m.info_results[self.rid] = {}
            for (pluginname, urls) in plugins.items():
                plugin = self.pyload.plugin_manager.get_plugin(pluginname, True)
                if hasattr(plugin, 'get_info'):
                    self.fetch_for_plugin(pluginname, plugin, urls, self.update_result, True)
                    if self.cache:
                        self.update_result(pluginname, [], True)
                else:
                    result = [(url, 0, 3, url) for url in urls]
                    self.update_result(pluginname, result, True)
            self.m.info_results[self.rid]['ALL_INFO_FETCHED'] = {}
        self.m.timestamp = time.time() + timedelta(minutes=5).total_seconds()

    def update_db(self, plugin, result):
        if False:
            while True:
                i = 10
        self.pyload.files.update_file_info(result, self.pid)

    def update_result(self, plugin, result, force=False):
        if False:
            i = 10
            return i + 15
        self.cache.extend(result)
        if len(self.cache) >= 20 or force:
            tmp = [(name, (url, OnlineStatus(name, plugin, 'unknown', status, int(size)))) for (name, size, status, url) in self.cache]
            data = parse_names(tmp)
            result = {}
            for (k, v) in data.items():
                for (url, status) in v:
                    status.packagename = k
                    result[url] = status
            self.m.set_info_results(self.rid, result)
            self.cache = []

    def update_cache(self, plugin, result):
        if False:
            i = 10
            return i + 15
        self.cache.extend(result)

    def fetch_for_plugin(self, pluginname, plugin, urls, cb, err=None):
        if False:
            while True:
                i = 10
        try:
            result = []
            process = []
            for url in urls:
                if url in self.m.info_cache:
                    result.append(self.m.info_cache[url])
                else:
                    process.append(url)
            if result:
                self.pyload.log.debug(f'Fetched {len(result)} values from cache for {pluginname}')
                cb(pluginname, result)
            if process:
                self.pyload.log.debug(f'Run Info Fetching for {pluginname}')
                for result in plugin.get_info(process):
                    if not isinstance(result, list):
                        result = [result]
                    for res in result:
                        self.m.info_cache[res[3]] = res
                    cb(pluginname, result)
            self.pyload.log.debug(f'Finished Info Fetching for {pluginname}')
        except Exception as exc:
            self.pyload.log.warning(self._('Info Fetching for {name} failed | {err}').format(name=pluginname, err=exc), exc_info=self.pyload.debug > 1, stack_info=self.pyload.debug > 2)
            if err:
                result = [(url, 0, 3, url) for url in urls]
                cb(pluginname, result)

    def decrypt_container(self, plugin, url):
        if False:
            i = 10
            return i + 15
        data = []
        self.pyload.log.debug(f'Pre-decrypting {url} with {plugin}')
        pyfile = PyFile(self.pyload.files, -1, url, url, 0, 0, '', plugin, -1, -1)
        pyfile.init_plugin()
        try:
            pyfile.plugin.setup()
            pyfile.plugin.load_to_disk()
            pyfile.plugin.decrypt(pyfile)
            pyfile.plugin.delete_tmp()
            for pack in pyfile.plugin.packages:
                pyfile.plugin.urls.extend(pack[1])
            data = self.pyload.plugin_manager.parse_urls(pyfile.plugin.urls)
            self.pyload.log.debug(f'Got {len(data)} links.')
        except Exception as exc:
            self.pyload.log.debug(f'Pre decrypting error: {exc}', exc_info=self.pyload.debug > 1, stack_info=self.pyload.debug > 2)
        finally:
            pyfile.release()
        return data