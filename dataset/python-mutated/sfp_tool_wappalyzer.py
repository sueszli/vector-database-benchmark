import os
import sys
import json
from subprocess import Popen, PIPE, TimeoutExpired
from spiderfoot import SpiderFootPlugin, SpiderFootEvent, SpiderFootHelpers

class sfp_tool_wappalyzer(SpiderFootPlugin):
    meta = {'name': 'Tool - Wappalyzer', 'summary': 'Wappalyzer indentifies technologies on websites.', 'flags': ['tool'], 'useCases': ['Footprint', 'Investigate'], 'categories': ['Content Analysis'], 'toolDetails': {'name': 'Wappalyzer', 'description': 'Wappalyzer identifies technologies on websites, including content management systems, ecommerce platforms, JavaScript frameworks, analytics tools and much more.', 'website': 'https://www.wappalyzer.com/', 'repository': 'https://github.com/AliasIO/Wappalyzer'}}
    opts = {'node_path': '/usr/bin/node', 'wappalyzer_path': ''}
    optdescs = {'node_path': 'Path to your NodeJS binary. Must be set.', 'wappalyzer_path': 'Path to your wappalyzer cli.js file. Must be set.'}
    results = None
    errorState = False

    def setup(self, sfc, userOpts=dict()):
        if False:
            while True:
                i = 10
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in userOpts.keys():
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['INTERNET_NAME']

    def producedEvents(self):
        if False:
            print('Hello World!')
        return ['OPERATING_SYSTEM', 'SOFTWARE_USED', 'WEBSERVER_TECHNOLOGY']

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if self.errorState:
            return
        if not self.opts['wappalyzer_path']:
            self.error('You enabled sfp_tool_wappalyzer but did not set a path to the tool!')
            self.errorState = True
            return
        exe = self.opts['wappalyzer_path']
        if self.opts['wappalyzer_path'].endswith('/'):
            exe = f'{exe}cli.js'
        if not os.path.isfile(exe):
            self.error(f'File does not exist: {exe}')
            self.errorState = True
            return
        if not SpiderFootHelpers.sanitiseInput(eventData):
            self.debug('Invalid input, skipping.')
            return
        if eventData in self.results:
            self.debug(f'Skipping {eventData} as already scanned.')
            return
        self.results[eventData] = True
        try:
            args = [self.opts['node_path'], exe, f'https://{eventData}']
            p = Popen(args, stdout=PIPE, stderr=PIPE)
            try:
                (stdout, stderr) = p.communicate(input=None, timeout=60)
                if p.returncode == 0:
                    content = stdout.decode(sys.stdin.encoding)
                else:
                    self.error('Unable to read Wappalyzer content.')
                    self.error(f'Error running Wappalyzer: {stderr}, {stdout}')
                    return
            except TimeoutExpired:
                p.kill()
                (stdout, stderr) = p.communicate()
                self.debug('Timed out waiting for Wappalyzer to finish')
                return
        except BaseException as e:
            self.error(f'Unable to run Wappalyzer: {e}')
            return
        try:
            data = json.loads(content)
            for item in data['technologies']:
                for cat in item['categories']:
                    if cat['name'] == 'Operating systems':
                        evt = SpiderFootEvent('OPERATING_SYSTEM', item['name'], self.__name__, event)
                    elif cat['name'] == 'Web servers':
                        evt = SpiderFootEvent('WEBSERVER_TECHNOLOGY', item['name'], self.__name__, event)
                    else:
                        evt = SpiderFootEvent('SOFTWARE_USED', item['name'], self.__name__, event)
                    self.notifyListeners(evt)
        except (KeyError, ValueError) as e:
            self.error(f"Couldn't parse the JSON output of Wappalyzer: {e}")
            self.error(f'Wappalyzer content: {content}')
            return