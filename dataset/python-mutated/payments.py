from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional
from ..checkout.fetch import CheckoutInfo, CheckoutLineInfo
if TYPE_CHECKING:
    from ..payment.interface import CustomerSource, GatewayResponse, PaymentData, PaymentGateway, TokenConfig

class PaymentInterface(ABC):

    @abstractmethod
    def list_payment_gateways(self, currency: Optional[str]=None, checkout_info: Optional['CheckoutInfo']=None, checkout_lines: Optional[Iterable['CheckoutLineInfo']]=None, channel_slug: Optional[str]=None, active_only: bool=True) -> list['PaymentGateway']:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def authorize_payment(self, gateway: str, payment_information: 'PaymentData', channel_slug: str) -> 'GatewayResponse':
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def capture_payment(self, gateway: str, payment_information: 'PaymentData', channel_slug: str) -> 'GatewayResponse':
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def refund_payment(self, gateway: str, payment_information: 'PaymentData', channel_slug: str) -> 'GatewayResponse':
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def void_payment(self, gateway: str, payment_information: 'PaymentData', channel_slug: str) -> 'GatewayResponse':
        if False:
            return 10
        pass

    @abstractmethod
    def confirm_payment(self, gateway: str, payment_information: 'PaymentData', channel_slug: str) -> 'GatewayResponse':
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def token_is_required_as_payment_input(self, gateway: str, channel_slug: str) -> bool:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def process_payment(self, gateway: str, payment_information: 'PaymentData', channel_slug: str) -> 'GatewayResponse':
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def get_client_token(self, gateway: str, token_config: 'TokenConfig', channel_slug: str) -> str:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def list_payment_sources(self, gateway: str, customer_id: str, channel_slug: str) -> list['CustomerSource']:
        if False:
            print('Hello World!')
        pass