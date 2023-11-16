from logging import Logger

class OAuthStateStore:

    @property
    def logger(self) -> Logger:
        if False:
            return 10
        raise NotImplementedError()

    def issue(self, *args, **kwargs) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def consume(self, state: str) -> bool:
        if False:
            return 10
        raise NotImplementedError()