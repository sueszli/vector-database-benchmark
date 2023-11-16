from abc import ABC, abstractmethod

class BaseLlm(ABC):

    @abstractmethod
    def chat_completion(self, prompt):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def get_source(self):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def get_api_key(self):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def get_model(self):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def get_models(self):
        if False:
            return 10
        pass

    @abstractmethod
    def verify_access_key(self):
        if False:
            for i in range(10):
                print('nop')
        pass