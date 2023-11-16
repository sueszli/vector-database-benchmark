from datetime import datetime
from typing import Any, Dict, Optional
import github.GithubObject
from github.GithubObject import Attribute, NotSet

class HookDeliverySummary(github.GithubObject.NonCompletableGithubObject):
    """
    This class represents a Summary of HookDeliveries
    """

    def _initAttributes(self) -> None:
        if False:
            while True:
                i = 10
        self._id: Attribute[int] = NotSet
        self._guid: Attribute[str] = NotSet
        self._delivered_at: Attribute[datetime] = NotSet
        self._redelivery: Attribute[bool] = NotSet
        self._duration: Attribute[float] = NotSet
        self._status: Attribute[str] = NotSet
        self._status_code: Attribute[int] = NotSet
        self._event: Attribute[str] = NotSet
        self._action: Attribute[str] = NotSet
        self._installation_id: Attribute[int] = NotSet
        self._repository_id: Attribute[int] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.get__repr__({'id': self._id.value})

    @property
    def id(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        return self._id.value

    @property
    def guid(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._guid.value

    @property
    def delivered_at(self) -> Optional[datetime]:
        if False:
            for i in range(10):
                print('nop')
        return self._delivered_at.value

    @property
    def redelivery(self) -> Optional[bool]:
        if False:
            return 10
        return self._redelivery.value

    @property
    def duration(self) -> Optional[float]:
        if False:
            while True:
                i = 10
        return self._duration.value

    @property
    def status(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._status.value

    @property
    def status_code(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        return self._status_code.value

    @property
    def event(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._event.value

    @property
    def action(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._action.value

    @property
    def installation_id(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        return self._installation_id.value

    @property
    def repository_id(self) -> Optional[int]:
        if False:
            print('Hello World!')
        return self._repository_id.value

    @property
    def url(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._url.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'guid' in attributes:
            self._guid = self._makeStringAttribute(attributes['guid'])
        if 'delivered_at' in attributes:
            self._delivered_at = self._makeDatetimeAttribute(attributes['delivered_at'])
        if 'redelivery' in attributes:
            self._redelivery = self._makeBoolAttribute(attributes['redelivery'])
        if 'duration' in attributes:
            self._duration = self._makeFloatAttribute(attributes['duration'])
        if 'status' in attributes:
            self._status = self._makeStringAttribute(attributes['status'])
        if 'status_code' in attributes:
            self._status_code = self._makeIntAttribute(attributes['status_code'])
        if 'event' in attributes:
            self._event = self._makeStringAttribute(attributes['event'])
        if 'action' in attributes:
            self._action = self._makeStringAttribute(attributes['action'])
        if 'installation_id' in attributes:
            self._installation_id = self._makeIntAttribute(attributes['installation_id'])
        if 'repository_id' in attributes:
            self._repository_id = self._makeIntAttribute(attributes['repository_id'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])

class HookDeliveryRequest(github.GithubObject.NonCompletableGithubObject):
    """
    This class represents a HookDeliveryRequest
    """

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._request_headers: Attribute[Dict] = NotSet
        self._payload: Attribute[Dict] = NotSet

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.get__repr__({'payload': self._payload.value})

    @property
    def headers(self) -> Optional[dict]:
        if False:
            while True:
                i = 10
        return self._request_headers.value

    @property
    def payload(self) -> Optional[dict]:
        if False:
            for i in range(10):
                print('nop')
        return self._payload.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'headers' in attributes:
            self._request_headers = self._makeDictAttribute(attributes['headers'])
        if 'payload' in attributes:
            self._payload = self._makeDictAttribute(attributes['payload'])

class HookDeliveryResponse(github.GithubObject.NonCompletableGithubObject):
    """
    This class represents a HookDeliveryResponse
    """

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.get__repr__({'payload': self._payload.value})

    @property
    def headers(self) -> Optional[dict]:
        if False:
            i = 10
            return i + 15
        return self._response_headers.value

    @property
    def payload(self) -> Optional[str]:
        if False:
            return 10
        return self._payload.value

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._response_headers: Attribute[Dict] = NotSet
        self._payload: Attribute[str] = NotSet

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            return 10
        if 'headers' in attributes:
            self._response_headers = self._makeDictAttribute(attributes['headers'])
        if 'payload' in attributes:
            self._payload = self._makeStringAttribute(attributes['payload'])

class HookDelivery(HookDeliverySummary):
    """
    This class represents a HookDelivery
    """

    def _initAttributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super()._initAttributes()
        self._request: Attribute[HookDeliveryRequest] = NotSet
        self._response: Attribute[HookDeliveryResponse] = NotSet

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return self.get__repr__({'id': self._id.value})

    @property
    def request(self) -> Optional[HookDeliveryRequest]:
        if False:
            i = 10
            return i + 15
        return self._request.value

    @property
    def response(self) -> Optional[HookDeliveryResponse]:
        if False:
            for i in range(10):
                print('nop')
        return self._response.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        super()._useAttributes(attributes)
        if 'request' in attributes:
            self._request = self._makeClassAttribute(HookDeliveryRequest, attributes['request'])
        if 'response' in attributes:
            self._response = self._makeClassAttribute(HookDeliveryResponse, attributes['response'])