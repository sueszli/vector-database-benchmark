"""
 Originally coded by @xtr4nge
"""
import threading
import logging
import json
import sys
from flask import Flask
from core.configwatcher import ConfigWatcher
from core.proxyplugins import ProxyPlugins
app = Flask(__name__)

class mitmfapi(ConfigWatcher):
    __shared_state = {}

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__dict__ = self.__shared_state
        self.host = self.config['MITMf']['MITMf-API']['host']
        self.port = int(self.config['MITMf']['MITMf-API']['port'])

    @app.route('/')
    def getPlugins():
        if False:
            print('Hello World!')
        pdict = {}
        for activated_plugin in ProxyPlugins().plugin_list:
            pdict[activated_plugin.name] = True
        for plugin in ProxyPlugins().all_plugins:
            if plugin.name not in pdict:
                pdict[plugin.name] = False
        return json.dumps(pdict)

    @app.route('/<plugin>')
    def getPluginStatus(plugin):
        if False:
            for i in range(10):
                print('nop')
        for p in ProxyPlugins().plugin_list:
            if plugin == p.name:
                return json.dumps('1')
        return json.dumps('0')

    @app.route('/<plugin>/<status>')
    def setPluginStatus(plugin, status):
        if False:
            return 10
        if status == '1':
            for p in ProxyPlugins().all_plugins:
                if p.name == plugin and p not in ProxyPlugins().plugin_list:
                    ProxyPlugins().add_plugin(p)
                    return json.dumps({'plugin': plugin, 'response': 'success'})
        elif status == '0':
            for p in ProxyPlugins().plugin_list:
                if p.name == plugin:
                    ProxyPlugins().remove_plugin(p)
                    return json.dumps({'plugin': plugin, 'response': 'success'})
        return json.dumps({'plugin': plugin, 'response': 'failed'})

    def startFlask(self):
        if False:
            return 10
        app.run(debug=False, host=self.host, port=self.port)

    def start(self):
        if False:
            return 10
        api_thread = threading.Thread(name='mitmfapi', target=self.startFlask)
        api_thread.setDaemon(True)
        api_thread.start()