import kivy
import socket
import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
kivy.require('1.9.0')
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
portNum = 1234

class MyRoot(BoxLayout):

    def _init_(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(MyRoot, self)._init_(**kwargs)
        Clock.schedule_once(self.connect_to_server)

    def send_message(self):
        if False:
            return 10
        client.send(f'{self.nickname_text.text}: {self.message_text.text}'.encode('utf-8'))

    def connect_to_server(self, *args):
        if False:
            i = 10
            return i + 15
        if self.nickname_text.text != '':
            print(f'IP ADDRESS YOU ENTERED: {self.ip_text.text}')
            client.connect((self.ip_text.text, portNum))
            message = client.recv(1024).decode('utf-8')
            if message == 'NICK':
                client.send(self.nickname_text.text.encode('utf-8'))
                self.send_btn.disabled = False
                self.message_text.disabled = False
                self.connect_btn.disabled = True
                self.ip_text.disabled = True
                self.make_invisible(self.connection_grid)
                self.make_invisible(self.connect_btn)
                threading.Thread(target=self.receive).start()

    def make_invisible(self, widget):
        if False:
            while True:
                i = 10
        widget.visible = False
        widget.size_hint_x = None
        widget.size_hint_y = None
        widget.height = 0
        widget.width = 0
        widget.text = ''
        widget.opacity = 0

    def receive(self):
        if False:
            for i in range(10):
                print('nop')
        stop = False
        while not stop:
            try:
                message = client.recv(1024).decode('utf-8')
                Clock.schedule_once(lambda dt: self.update_chat_text(message))
            except Exception as e:
                print('Error occurred:', str(e))
                client.close()
                stop = True

    def update_chat_text(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.chat_text.text += message + '\n'

class WebChatRoom(App):

    def build(self):
        if False:
            return 10
        return MyRoot()
webChatRoom = WebChatRoom()
webChatRoom.run()