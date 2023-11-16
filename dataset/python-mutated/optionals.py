from builtins import _test_sink, _test_source
from typing import Optional

class Client:

    def offer(self, message):
        if False:
            i = 10
            return i + 15
        _test_sink(message)

class ClientSingleton:

    def get_instance(self) -> Optional[Client]:
        if False:
            print('Hello World!')
        return Client()
client: ClientSingleton = ClientSingleton()

def test():
    if False:
        print('Hello World!')
    client.get_instance().offer(_test_source())