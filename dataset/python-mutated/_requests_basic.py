import collections.abc as collections
from requests.structures import CaseInsensitiveDict
from ._http_response_impl import _HttpResponseBaseImpl, HttpResponseImpl, _HttpResponseBackcompatMixinBase
from ..pipeline.transport._requests_basic import StreamDownloadGenerator

class _ItemsView(collections.ItemsView):

    def __contains__(self, item):
        if False:
            return 10
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            return False
        for (k, v) in self.__iter__():
            if item[0].lower() == k.lower() and item[1] == v:
                return True
        return False

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'ItemsView({})'.format(dict(self.__iter__()))

class _CaseInsensitiveDict(CaseInsensitiveDict):
    """Overriding default requests dict so we can unify
    to not raise if users pass in incorrect items to contains.
    Instead, we return False
    """

    def items(self):
        if False:
            i = 10
            return i + 15
        "Return a new view of the dictionary's items.\n\n        :rtype: ~collections.abc.ItemsView[str, str]\n        :returns: a view object that displays a list of (key, value) tuple pairs\n        "
        return _ItemsView(self)

class _RestRequestsTransportResponseBaseMixin(_HttpResponseBackcompatMixinBase):
    """Backcompat mixin for the sync and async requests responses

    Overriding the default mixin behavior here because we need to synchronously
    read the response's content for the async requests responses
    """

    def _body(self):
        if False:
            for i in range(10):
                print('nop')
        if self._content is None:
            self._content = self._internal_response.content
        return self._content

class _RestRequestsTransportResponseBase(_HttpResponseBaseImpl, _RestRequestsTransportResponseBaseMixin):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        internal_response = kwargs.pop('internal_response')
        content = None
        if internal_response._content_consumed:
            content = internal_response.content
        headers = _CaseInsensitiveDict(internal_response.headers)
        super(_RestRequestsTransportResponseBase, self).__init__(internal_response=internal_response, status_code=internal_response.status_code, headers=headers, reason=internal_response.reason, content_type=headers.get('content-type'), content=content, **kwargs)

class RestRequestsTransportResponse(HttpResponseImpl, _RestRequestsTransportResponseBase):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(RestRequestsTransportResponse, self).__init__(stream_download_generator=StreamDownloadGenerator, **kwargs)