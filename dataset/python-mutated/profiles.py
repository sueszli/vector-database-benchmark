from typing import Any, Iterable, List, Mapping
import requests
from airbyte_cdk.models import SyncMode
from source_amazon_ads.schemas import Profile
from source_amazon_ads.streams.common import AmazonAdsStream

class Profiles(AmazonAdsStream):
    """
    This stream corresponds to Amazon Advertising API - Profiles
    https://advertising.amazon.com/API/docs/en-us/reference/2/profiles#/Profiles
    """
    primary_key = 'profileId'
    model = Profile

    def path(self, **kvargs) -> str:
        if False:
            while True:
                i = 10
        return 'v2/profiles?profileTypeFilter=seller,vendor'

    def parse_response(self, response: requests.Response, **kwargs) -> Iterable[Mapping]:
        if False:
            print('Hello World!')
        for record in super().parse_response(response, **kwargs):
            profile_id_obj = self.model.parse_obj(record)
            self._profiles.append(profile_id_obj)
            yield record

    def read_records(self, *args, **kvargs) -> Iterable[Mapping[str, Any]]:
        if False:
            i = 10
            return i + 15
        if self._profiles:
            yield from [profile.dict(exclude_unset=True) for profile in self._profiles]
        else:
            yield from super().read_records(*args, **kvargs)

    def get_all_profiles(self) -> List[Profile]:
        if False:
            return 10
        '\n        Fetch all profiles and return it as list. We need this to set\n        dependecies for other streams since all of the Amazon Ads API calls\n        require profile id to be passed.\n        :return List of profile object\n        '
        return [self.model.parse_obj(profile) for profile in self.read_records(SyncMode.full_refresh)]