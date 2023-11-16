import json
import os
from jadi import component
import aj
from aj.api.http import get, HttpPlugin
from aj.plugins import PluginManager
from aj.api.endpoint import endpoint

@component(HttpPlugin)
class ResourcesHandler(HttpPlugin):

    def __init__(self, http_context):
        if False:
            for i in range(10):
                print('nop')
        self.cache = {}
        self.use_cache = not aj.debug
        self.mgr = PluginManager.get(aj.context)

    def __wrap_js(self, name, js):
        if False:
            return 10
        '\n        Wrap the content with exception handler.\n\n        :param name: File path\n        :type name: string\n        :param js: Content of the resource\n        :type js: string\n        :return: Wrapped content\n        :rtype: string\n        '
        return f"\n            try {{\n                {js}\n            }} catch (err) {{\n                console.warn('Plugin load error:');\n                console.warn(' * {name}');\n                console.error('  ', err);\n            }}\n        "

    @get('/resources/all\\.(?P<group>.+)')
    @endpoint(page=True, auth=False)
    def handle_build(self, http_context, group=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deliver all extern resources for the current page.\n\n        :param http_context: HttpContext\n        :type http_context: HttpContext\n        :param group: File extension/type, e.g. css, js ...\n        :type group: string\n        :return: Compressed content with gzip\n        :rtype: gzip\n        '
        if self.use_cache and group in self.cache:
            content = self.cache[group]
        else:
            content = ''
            if group in ['js', 'css', 'vendor.js', 'vendor.css']:
                for plugin in self.mgr:
                    path = self.mgr.get_content_path(plugin, f'resources/build/all.{group}')
                    if os.path.exists(path):
                        with open(path, encoding='utf-8') as f:
                            file_content = f.read()
                        if group == 'js':
                            file_content = self.__wrap_js(path, file_content)
                        content += file_content
            if group == 'init.js':
                ng_modules = {}
                for plugin in self.mgr:
                    for resource in self.mgr[plugin]['info']['resources']:
                        if resource['path'].startswith('ng:'):
                            ng_modules.setdefault(plugin, []).append(resource['path'].split(':')[-1])
                content = f'\n                    window.__ngModules = {json.dumps(ng_modules)};\n                '
            if group == 'locale.js':
                lang = http_context.query.get('lang', None)
                if lang:
                    js_locale = {}
                    for plugin in self.mgr:
                        locale_dir = self.mgr.get_content_path(plugin, 'locale')
                        js_path = os.path.join(locale_dir, lang, 'LC_MESSAGES', 'app.js')
                        if os.path.exists(js_path):
                            with open(js_path, encoding='utf-8') as j:
                                js_locale.update(json.load(j))
                    content = json.dumps(js_locale)
                else:
                    content = ''
            if group == 'partials.js':
                content = '\n                    angular.module("core.templates", []);\n                    angular.module("core.templates").run(\n                        ["$templateCache", function($templateCache) {\n                '
                for plugin in self.mgr:
                    for resource in self.mgr[plugin]['info']['resources']:
                        path = resource['path']
                        name = resource.get('overrides', f'{plugin}:{path}')
                        if name.endswith('.html'):
                            path = self.mgr.get_content_path(plugin, path)
                            if os.path.exists(path):
                                with open(path, encoding='utf-8') as t:
                                    template = t.read()
                                content += f'\n                                      $templateCache.put("{http_context.prefix}/{name}", {json.dumps(template)});\n                                '
                content += '\n                    }]);\n                '
            self.cache[group] = content
        http_context.add_header('Content-Type', {'css': 'text/css', 'js': 'application/javascript; charset=utf-8', 'vendor.css': 'text/css', 'vendor.js': 'application/javascript; charset=utf-8', 'init.js': 'application/javascript; charset=utf-8', 'locale.js': 'application/javascript; charset=utf-8', 'partials.js': 'application/javascript; charset=utf-8'}[group])
        http_context.respond_ok()
        return http_context.gzip(content=content.encode('utf-8'))

    @get('/resources/(?P<plugin>\\w+)/(?P<path>.+)')
    @endpoint(page=True, auth=False)
    def handle_file(self, http_context, plugin=None, path=None):
        if False:
            print('Hello World!')
        '\n        Connector to get a specific file from plugin.\n\n        :param http_context: HttpContext\n        :type http_context: HttpContext\n        :param plugin: Plugin name\n        :type plugin: string\n        :param path: Path of the file\n        :type path: string\n        :return: Compressed content of the file\n        :rtype: gzip\n        '
        if '..' in path:
            return http_context.respond_not_found()
        return http_context.file(PluginManager.get(aj.context).get_content_path(plugin, path))