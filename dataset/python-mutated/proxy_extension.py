import zipfile
import os

def create_proxy_extension(proxy):
    if False:
        print('Hello World!')
    'takes proxy looks like login:password@ip:port'
    ip = proxy.split('@')[1].split(':')[0]
    port = int(proxy.split(':')[-1])
    login = proxy.split(':')[0]
    password = proxy.split('@')[0].split(':')[1]
    manifest_json = '\n        {\n            "version": "1.0.0",\n            "manifest_version": 2,\n            "name": "Chrome Proxy",\n            "permissions": [\n                "proxy",\n                "tabs",\n                "unlimitedStorage",\n                "storage",\n                "<all_urls>",\n                "webRequest",\n                "webRequestBlocking"\n            ],\n            "background": {\n                "scripts": ["background.js"]\n            },\n            "minimum_chrome_version":"22.0.0"\n        }\n    '
    background_js = '\n        var config = {\n                mode: "fixed_servers",\n                rules: {\n                  singleProxy: {\n                    scheme: "http",\n                    host: "%s",\n                    port: parseInt(%s)\n                  },\n                  bypassList: ["localhost"]\n                }\n              };\n        chrome.proxy.settings.set({value: config, scope: "regular"}, \n        function() {});\n        function callbackFn(details) {\n            return {\n                authCredentials: {\n                    username: "%s",\n                    password: "%s"\n                }\n            };\n        }\n        chrome.webRequest.onAuthRequired.addListener(\n                    callbackFn,\n                    {urls: ["<all_urls>"]},\n                    [\'blocking\']\n        );\n    ' % (ip, port, login, password)
    dir_path = 'assets/chrome_extensions'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    pluginfile = '%s/proxy_auth_%s:%s.zip' % (dir_path, ip, port)
    with zipfile.ZipFile(pluginfile, 'w') as zp:
        zp.writestr('manifest.json', manifest_json)
        zp.writestr('background.js', background_js)
    return pluginfile