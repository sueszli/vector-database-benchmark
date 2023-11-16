from __future__ import unicode_literals
from kivy.support import install_twisted_reactor
install_twisted_reactor()
from twisted.internet import reactor, protocol

class EchoClient(protocol.Protocol):

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        self.factory.app.on_connection(self.transport)

    def dataReceived(self, data):
        if False:
            while True:
                i = 10
        self.factory.app.print_message(data.decode('utf-8'))

class EchoClientFactory(protocol.ClientFactory):
    protocol = EchoClient

    def __init__(self, app):
        if False:
            return 10
        self.app = app

    def startedConnecting(self, connector):
        if False:
            while True:
                i = 10
        self.app.print_message('Started to connect.')

    def clientConnectionLost(self, connector, reason):
        if False:
            i = 10
            return i + 15
        self.app.print_message('Lost connection.')

    def clientConnectionFailed(self, connector, reason):
        if False:
            print('Hello World!')
        self.app.print_message('Connection failed.')
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout

class TwistedClientApp(App):
    connection = None
    textbox = None
    label = None

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        root = self.setup_gui()
        self.connect_to_server()
        return root

    def setup_gui(self):
        if False:
            return 10
        self.textbox = TextInput(size_hint_y=0.1, multiline=False)
        self.textbox.bind(on_text_validate=self.send_message)
        self.label = Label(text='connecting...\n')
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.label)
        layout.add_widget(self.textbox)
        return layout

    def connect_to_server(self):
        if False:
            return 10
        reactor.connectTCP('localhost', 8000, EchoClientFactory(self))

    def on_connection(self, connection):
        if False:
            while True:
                i = 10
        self.print_message('Connected successfully!')
        self.connection = connection

    def send_message(self, *args):
        if False:
            return 10
        msg = self.textbox.text
        if msg and self.connection:
            self.connection.write(msg.encode('utf-8'))
            self.textbox.text = ''

    def print_message(self, msg):
        if False:
            print('Hello World!')
        self.label.text += '{}\n'.format(msg)
if __name__ == '__main__':
    TwistedClientApp().run()