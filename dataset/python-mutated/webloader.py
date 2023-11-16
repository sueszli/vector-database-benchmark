from __future__ import print_function
import renpy
import os
import renpy.display
import threading
import time
queue = []
to_unlink = {}
queue_lock = threading.RLock()
if renpy.emscripten:
    import emscripten, json
    emscripten.run_script('RenPyWeb = {\n    xhr_id: 0,\n    xhrs: {},\n\n    dl_new: function(path) {\n        var xhr = new XMLHttpRequest();\n        xhr.responseType = \'arraybuffer\';\n        xhr.onerror = function() {\n            console.log("Network error", xhr);\n        };\n        xhr.onload = function() {\n            if (xhr.status == 200 || xhr.status == 304 || xhr.status == 206 || (xhr.status == 0 && xhr.response)) {\n                // Create file reusing XHR\'s buffer (no-copy)\n                try { FS.unlink(path); } catch(error) {}\n                FS.writeFile(path, new Uint8Array(xhr.response), {canOwn:true});\n            } else {\n                console.log("Download error", xhr);\n            }\n        };\n        xhr.open(\'GET\', path);\n        xhr.send();\n        RenPyWeb.xhrs[RenPyWeb.xhr_id] = xhr;\n        var ret = RenPyWeb.xhr_id;\n        RenPyWeb.xhr_id++;\n        return ret;\n    },\n\n    dl_get: function(xhr_id) {\n        return RenPyWeb.xhrs[xhr_id];\n    },\n\n    dl_free: function(xhr_id) {\n        delete RenPyWeb.xhrs[xhr_id];\n        // Note: xhr.response kept alive until file is unlinked\n    },\n};\n')

    class XMLHttpRequest(object):

        def __init__(self, filename):
            if False:
                print('Hello World!')
            url = 'game/' + filename
            self.id = emscripten.run_script_int('RenPyWeb.dl_new({})'.format(json.dumps(url)))

        def __del__(self):
            if False:
                print('Hello World!')
            emscripten.run_script('RenPyWeb.dl_free({});'.format(self.id))

        @property
        def readyState(self):
            if False:
                i = 10
                return i + 15
            return emscripten.run_script_int('RenPyWeb.dl_get({}).readyState'.format(self.id))

        @property
        def status(self):
            if False:
                while True:
                    i = 10
            return emscripten.run_script_int('RenPyWeb.dl_get({}).status'.format(self.id))

        @property
        def statusText(self):
            if False:
                i = 10
                return i + 15
            return emscripten.run_script_string('RenPyWeb.dl_get({}).statusText'.format(self.id))
elif os.environ.get('RENPY_SIMULATE_DOWNLOAD', False):
    import urllib, urllib.parse, random, requests

    class XMLHttpRequest(object):

        def __init__(self, filename):
            if False:
                for i in range(10):
                    print('nop')
            self.done = False
            self.error = None
            url = 'http://127.0.0.1:8042/game/' + urllib.parse.quote(filename)

            def thread_main():
                if False:
                    return 10
                try:
                    time.sleep(random.random() * 0.5)
                    r = requests.get(url)
                    fullpath = os.path.join(renpy.config.gamedir, filename)
                    with queue_lock:
                        with open(fullpath, 'wb') as f:
                            f.write(r.content)
                except requests.RequestException as e:
                    self.error = repr(e)
                except Exception as e:
                    self.error = 'Error: ' + str(e)
                self.done = True
            threading.Thread(target=thread_main, name='XMLHttpRequest').start()

        @property
        def readyState(self):
            if False:
                return 10
            if self.done:
                return 4
            else:
                return 0

        @property
        def status(self):
            if False:
                return 10
            if self.error:
                return 0
            return 200

        @property
        def statusText(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.error or 'OK'

class DownloadNeeded(Exception):

    def __init__(self, relpath, rtype, size):
        if False:
            return 10
        self.relpath = relpath
        self.rtype = rtype
        self.size = size

class ReloadRequest:

    def __init__(self, relpath, rtype, data):
        if False:
            while True:
                i = 10
        self.relpath = relpath
        self.rtype = rtype
        self.data = data
        self.gc_gen = 0
        self.xhr = XMLHttpRequest(self.relpath)

    def download_completed(self):
        if False:
            print('Hello World!')
        return self.xhr.readyState == 4

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return u"<ReloadRequest {} '{}' {}>".format(self.rtype, self.relpath, self.download_completed())

def enqueue(relpath, rtype, data):
    if False:
        print('Hello World!')
    global queue
    with queue_lock:
        voice_count = 0
        for rr in queue:
            if rr.rtype == rtype == 'image':
                image_filename = data
                if rr.data == image_filename:
                    return
            elif rr.rtype == rtype == 'music' and rr.relpath == relpath:
                return
            elif rr.rtype == rtype == 'voice':
                if rr.relpath == relpath:
                    return
                voice_count += 1
        if voice_count > renpy.config.predict_statements:
            return
        queue.append(ReloadRequest(relpath, rtype, data))

def process_downloaded_resources():
    if False:
        for i in range(10):
            print('nop')
    global queue
    if not queue:
        return
    with queue_lock:
        todo = queue[:]
        postponed = []
        try:
            while todo:
                rr = todo.pop()
                if not rr.download_completed():
                    postponed.append(rr)
                    continue
                if rr.rtype == 'image':
                    fullpath = os.path.join(renpy.config.gamedir, rr.relpath)
                    if not os.path.exists(fullpath):
                        raise IOError("Download error: {} ('{}' > '{}')".format(rr.xhr.statusText or 'network error', rr.relpath, fullpath))
                    image_filename = rr.data
                    renpy.exports.flush_cache_file(image_filename)
                    fullpath = os.path.join(renpy.config.gamedir, rr.relpath)
                    to_unlink[fullpath] = time.time()
                elif rr.rtype == 'music':
                    pass
                elif rr.rtype == 'voice':
                    fullpath = os.path.join(renpy.config.gamedir, rr.relpath)
                    to_unlink[fullpath] = time.time() + 120
        finally:
            queue = postponed + todo
    ttl = 60
    current_time = time.time()
    for (fullpath, value) in tuple(to_unlink.items()):
        delta = current_time - value
        if delta > ttl:
            os.unlink(fullpath)
            del to_unlink[fullpath]