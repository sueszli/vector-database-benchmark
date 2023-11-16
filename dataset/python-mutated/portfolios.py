from typing import Any, Iterable, Mapping, MutableMapping
from source_amazon_ads.schemas import Portfolio
from source_amazon_ads.streams.common import AmazonAdsStream

class Portfolios(AmazonAdsStream):
    """
    This stream corresponds to Amazon Advertising API - Portfolios
    https://advertising.amazon.com/API/docs/en-us/reference/2/portfolios
    """
    primary_key = 'portfolioId'
    model = Portfolio

    def path(self, **kvargs) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'v2/portfolios/extended'

    def read_records(self, *args, **kvargs) -> Iterable[Mapping[str, Any]]:
        if False:
            return 10
        '\n        Iterate through self._profiles list and send read all records for each profile.\n        '
        for profile in self._profiles:
            self._current_profile_id = profile.profileId
            yield from super().read_records(*args, **kvargs)

    def request_headers(self, *args, **kvargs) -> MutableMapping[str, Any]:
        if False:
            print('Hello World!')
        headers = super().request_headers(*args, **kvargs)
        headers['Amazon-Advertising-API-Scope'] = str(self._current_profile_id)
        return headers