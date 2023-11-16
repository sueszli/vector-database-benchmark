from abc import ABC, abstractmethod

class BaseImageLlm(ABC):

    @abstractmethod
    def get_image_model(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def generate_image(self, prompt: str, size: int=512, num: int=2):
        if False:
            print('Hello World!')
        pass