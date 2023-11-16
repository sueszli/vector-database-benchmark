"""
Interface for MapperConsumerFactory, Producer, Mapper, ListInfoPullerConsumer
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')

class ListInfoPullerConsumer(ABC, Generic[InputType]):
    """
    Interface definition to consume and display data
    """

    @abstractmethod
    def consume(self, data: InputType):
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        data: TypeVar\n            Data for the consumer to print\n        '

class Mapper(ABC, Generic[InputType, OutputType]):
    """
    Interface definition to map data to json or table
    """

    @abstractmethod
    def map(self, data: InputType) -> OutputType:
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        data: TypeVar\n            Data for the mapper to map\n\n        Returns\n        -------\n        Any\n            Mapped output given the data\n        '

class Producer(ABC):
    """
    Interface definition to produce data for the mappers and consumers
    """
    mapper: Mapper
    consumer: ListInfoPullerConsumer

    @abstractmethod
    def produce(self):
        if False:
            return 10
        '\n        Produces the data for the mappers and consumers\n        '

class MapperConsumerFactoryInterface(ABC):
    """
    Interface definition to create mapper-consumer factories
    """

    @abstractmethod
    def create(self, producer, output):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        producer: str\n            A string indicating which producer is calling the function\n        output: str\n            A string indicating the output type\n\n        Returns\n        -------\n        MapperConsumerContainer\n            A container that contains a mapper and a consumer\n        '

class ProducersEnum(Enum):
    STACK_OUTPUTS_PRODUCER = 1
    RESOURCES_PRODUCER = 2
    ENDPOINTS_PRODUCER = 3