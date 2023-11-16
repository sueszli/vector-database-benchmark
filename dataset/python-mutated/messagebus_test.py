"""This is a unittest for the message buss

It's important to note that this requires this test to run mycroft service
to test the buss.  It is not expected that the service be already running
when the tests are ran.
"""
import time
import unittest
from subprocess import Popen, call
from threading import Thread
from mycroft.messagebus.client import MessageBusClient
from mycroft.messagebus.message import Message

class TestMessagebusMethods(unittest.TestCase):
    """This class is for testing the messsagebus.

    It currently only tests send and receive.  The tests could include
    more.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This sets up for testing the message buss\n\n        This requires starting the mycroft service and creating two\n        WebsocketClient object to talk with eachother.  Not this is\n        threaded and will require cleanup\n        '
        self.pid = Popen(['python3', '-m', 'mycroft.messagebus.service']).pid
        self.ws1 = MessageBusClient()
        self.ws2 = MessageBusClient()
        self.handle1 = False
        self.handle2 = False
        Thread(target=self.ws1.run_forever).start()
        Thread(target=self.ws2.run_forever).start()
        self.ws1.on('ws1.message', self.onHandle1)
        self.ws2.on('ws2.message', self.onHandle2)

    def onHandle1(self, event):
        if False:
            i = 10
            return i + 15
        'This is the handler for ws1.message\n\n        This for now simply sets a flag to true when received.\n\n        Args:\n            event(Message): this is the message received\n        '
        self.handle1 = True

    def onHandle2(self, event):
        if False:
            while True:
                i = 10
        'This is the handler for ws2.message\n\n        This for now simply sets a flag to true when received.\n\n        Args:\n            event(Message): this is the message received\n        '
        self.handle2 = True

    def tearDown(self):
        if False:
            while True:
                i = 10
        'This is the clean up for the tests\n\n        This will close the websockets ending the threads then kill the\n        mycroft service that was started in setUp.\n        '
        self.ws1.close()
        self.ws2.close()
        retcode = call(['kill', '-9', str(self.pid)])

    def test_ClientServer(self):
        if False:
            return 10
        'This is the test to send a message from each of the websockets\n        to the other.\n        '
        self.ws2.emit(Message('ws1.message'))
        self.ws1.emit(Message('ws2.message'))
        time.sleep(0.2)
        self.assertTrue(self.handle1)
        self.assertTrue(self.handle2)
if __name__ == '__main__':
    'This is to start the testing'
    unittest.main()