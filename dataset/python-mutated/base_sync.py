from abc import ABC, abstractmethod

class BaseSync(ABC):

    @abstractmethod
    def sync_data(self) -> None:
        if False:
            while True:
                i = 10
        pass