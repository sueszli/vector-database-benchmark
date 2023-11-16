import random
import sys
import threading
import time
import webview
'\nCreate an application without a HTTP server. The application uses Javascript API object to communicate between Python and Javascript.\n'
html = '\n<!DOCTYPE html>\n<html>\n<head lang="en">\n<meta charset="UTF-8">\n\n<style>\n    #response-container {\n        display: none;\n        padding: 3rem;\n        margin: 3rem 5rem;\n        font-size: 120%;\n        border: 5px dashed #ccc;\n    }\n\n    label {\n        margin-left: 0.3rem;\n        margin-right: 0.3rem;\n    }\n\n    button {\n        font-size: 100%;\n        padding: 0.5rem;\n        margin: 0.3rem;\n        text-transform: uppercase;\n    }\n\n</style>\n</head>\n<body>\n\n\n<h1>JS API Example</h1>\n<p id=\'pywebview-status\'><i>pywebview</i> is not ready</p>\n\n<button onClick="initialize()">Hello Python</button><br/>\n<button id="heavy-stuff-btn" onClick="doHeavyStuff()">Perform a heavy operation</button><br/>\n<button onClick="getRandomNumber()">Get a random number</button><br/>\n<label for="name_input">Say hello to:</label><input id="name_input" placeholder="put a name here">\n<button onClick="greet()">Greet</button><br/>\n<button onClick="catchException()">Catch Exception</button><br/>\n\n\n<div id="response-container"></div>\n<script>\n    window.addEventListener(\'pywebviewready\', function() {\n        var container = document.getElementById(\'pywebview-status\')\n        container.innerHTML = \'<i>pywebview</i> is ready\'\n    })\n\n    function showResponse(response) {\n        var container = document.getElementById(\'response-container\')\n\n        container.innerText = response.message\n        container.style.display = \'block\'\n    }\n\n    function initialize() {\n        pywebview.api.init().then(showResponse)\n    }\n\n    function doHeavyStuff() {\n        var btn = document.getElementById(\'heavy-stuff-btn\')\n\n        pywebview.api.doHeavyStuff().then(function(response) {\n            showResponse(response)\n            btn.onclick = doHeavyStuff\n            btn.innerText = \'Perform a heavy operation\'\n        })\n\n        showResponse({message: \'Working...\'})\n        btn.innerText = \'Cancel the heavy operation\'\n        btn.onclick = cancelHeavyStuff\n    }\n\n    function cancelHeavyStuff() {\n        pywebview.api.cancelHeavyStuff()\n    }\n\n    function getRandomNumber() {\n        pywebview.api.getRandomNumber().then(showResponse)\n    }\n\n    function greet() {\n        var name_input = document.getElementById(\'name_input\').value;\n        pywebview.api.sayHelloTo(name_input).then(showResponse)\n    }\n\n    function catchException() {\n        pywebview.api.error().catch(showResponse)\n    }\n\n</script>\n</body>\n</html>\n'

class Api:

    def __init__(self):
        if False:
            print('Hello World!')
        self.cancel_heavy_stuff_flag = False

    def init(self):
        if False:
            i = 10
            return i + 15
        response = {'message': 'Hello from Python {0}'.format(sys.version)}
        return response

    def getRandomNumber(self):
        if False:
            i = 10
            return i + 15
        response = {'message': 'Here is a random number courtesy of randint: {0}'.format(random.randint(0, 100000000))}
        return response

    def doHeavyStuff(self):
        if False:
            print('Hello World!')
        time.sleep(0.1)
        now = time.time()
        self.cancel_heavy_stuff_flag = False
        for i in range(0, 1000000):
            _ = i * random.randint(0, 1000)
            if self.cancel_heavy_stuff_flag:
                response = {'message': 'Operation cancelled'}
                break
        else:
            then = time.time()
            response = {'message': 'Operation took {0:.1f} seconds on the thread {1}'.format(then - now, threading.current_thread())}
        return response

    def cancelHeavyStuff(self):
        if False:
            while True:
                i = 10
        time.sleep(0.1)
        self.cancel_heavy_stuff_flag = True

    def sayHelloTo(self, name):
        if False:
            return 10
        response = {'message': 'Hello {0}!'.format(name)}
        return response

    def error(self):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('This is a Python exception')
if __name__ == '__main__':
    api = Api()
    window = webview.create_window('JS API example', html=html, js_api=api)
    webview.start()