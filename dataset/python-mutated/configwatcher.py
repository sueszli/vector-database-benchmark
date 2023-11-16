import pyinotify
import threading
from configobj import ConfigObj

class ConfigWatcher(pyinotify.ProcessEvent, object):

    @property
    def config(self):
        if False:
            for i in range(10):
                print('nop')
        return ConfigObj('./config/mitmf.conf')

    def process_IN_MODIFY(self, event):
        if False:
            i = 10
            return i + 15
        self.on_config_change()

    def start_config_watch(self):
        if False:
            return 10
        wm = pyinotify.WatchManager()
        wm.add_watch('./config/mitmf.conf', pyinotify.IN_MODIFY)
        notifier = pyinotify.Notifier(wm, self)
        t = threading.Thread(name='ConfigWatcher', target=notifier.loop)
        t.setDaemon(True)
        t.start()

    def on_config_change(self):
        if False:
            print('Hello World!')
        ' We can subclass this function to do stuff after the config file has been modified'
        pass