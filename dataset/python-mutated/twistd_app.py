from kivy.support import install_twisted_reactor
install_twisted_reactor()
import os
import sys
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.properties import BooleanProperty
from kivy.lang import Builder
from twisted.scripts._twistd_unix import UnixApplicationRunner, ServerOptions
from twisted.application.service import IServiceCollection
TWISTD = 'twistd web --listen=tcp:8087'

class AndroidApplicationRunner(UnixApplicationRunner):

    def run(self):
        if False:
            while True:
                i = 10
        self.preApplication()
        self.application = self.createOrGetApplication()
        self.logger.start(self.application)
        sc = IServiceCollection(self.application)
        sc.startService()
        return self.application
Builder.load_string("\n<TwistedTwistd>:\n    cols: 1\n    Button:\n        text: root.running and 'STOP' or 'START'\n        on_release: root.cb_twistd()\n")

class TwistedTwistd(GridLayout):
    running = BooleanProperty(False)

    def cb_twistd(self, *la):
        if False:
            print('Hello World!')
        if self.running:
            IServiceCollection(self.app).stopService()
            self.running = False
        else:
            sys.path.insert(0, os.path.abspath(os.getcwd()))
            sys.argv = TWISTD.split(' ')
            config = ServerOptions()
            config.parseOptions()
            self.app = AndroidApplicationRunner(config).run()
            self.running = True

class TwistedTwistdApp(App):

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        return TwistedTwistd()
if __name__ == '__main__':
    TwistedTwistdApp().run()