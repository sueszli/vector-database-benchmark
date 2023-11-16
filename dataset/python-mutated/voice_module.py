from abc import ABC, abstractmethod

class VoiceModule(ABC):

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def update_usage(self):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def get_remaining_characters(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def generate_voice(self, text, outputfile):
        if False:
            while True:
                i = 10
        pass