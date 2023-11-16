from typing import Any
from typing import ClassVar
from typing import Optional
from typing import Type
from typing import Union
from ...serde.serializable import serializable
from ..response import SyftError
from ..response import SyftSuccess

@serializable()
class QueueClientConfig:
    pass

@serializable()
class AbstractMessageHandler:
    queue_name: ClassVar[str]

    @staticmethod
    def handle_message(message: bytes):
        if False:
            print('Hello World!')
        raise NotImplementedError

@serializable(attrs=['message_handler', 'queue_name', 'address'])
class QueueConsumer:
    message_handler: AbstractMessageHandler
    queue_name: str
    address: str

    def receive(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def run(self) -> None:
        if False:
            return 10
        raise NotImplementedError

    def close(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

@serializable()
class QueueProducer:
    address: str
    queue_name: str

    def send(self, message: Any):
        if False:
            return 10
        raise NotImplementedError

    def close(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

@serializable()
class QueueClient:
    pass

@serializable()
class QueueConfig:
    """Base Queue configuration"""
    client_type: Type[QueueClient]
    client_config: QueueClientConfig

@serializable()
class BaseQueueManager:
    config: QueueConfig

    def __init__(self, config: QueueConfig):
        if False:
            while True:
                i = 10
        self.config = config
        self.post_init()

    def post_init(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def create_consumer(self, message_handler: Type[AbstractMessageHandler], address: Optional[str]) -> QueueConsumer:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def create_producer(self, queue_name: str) -> QueueProducer:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def send(self, message: bytes, queue_name: str) -> Union[SyftSuccess, SyftError]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @property
    def publisher(self) -> QueueProducer:
        if False:
            return 10
        raise NotImplementedError