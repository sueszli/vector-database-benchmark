from kivy.support import install_twisted_reactor
install_twisted_reactor()
from twisted.internet import reactor
from twisted.internet import protocol

class EchoServer(protocol.Protocol):

    def dataReceived(self, data):
        if False:
            print('Hello World!')
        response = self.factory.app.handle_message(data)
        if response:
            self.transport.write(response)

class EchoServerFactory(protocol.Factory):
    protocol = EchoServer

    def __init__(self, app):
        if False:
            for i in range(10):
                print('nop')
        self.app = app
from kivy.app import App
from kivy.uix.label import Label

class TwistedServerApp(App):
    label = None

    def build(self):
        if False:
            while True:
                i = 10
        self.label = Label(text='server started\n')
        reactor.listenTCP(8000, EchoServerFactory(self))
        return self.label

    def handle_message(self, msg):
        if False:
            print('Hello World!')
        msg = msg.decode('utf-8')
        self.label.text = 'received:  {}\n'.format(msg)
        if msg == 'ping':
            msg = 'Pong'
        if msg == 'plop':
            msg = 'Kivy Rocks!!!'
        self.label.text += 'responded: {}\n'.format(msg)
        return msg.encode('utf-8')
if __name__ == '__main__':
    TwistedServerApp().run()