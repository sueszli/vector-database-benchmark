"""Specific data provider for the USA (en)."""
from mimesis.locales import Locale
from mimesis.providers import BaseDataProvider
from mimesis.types import MissingSeed, Seed
__all__ = ['USASpecProvider']

class USASpecProvider(BaseDataProvider):
    """Class that provides special data for the USA (en)."""

    def __init__(self, seed: Seed=MissingSeed) -> None:
        if False:
            return 10
        'Initialize attributes.'
        super().__init__(locale=Locale.EN, seed=seed)

    class Meta:
        name = 'usa_provider'
        datafile = None

    def tracking_number(self, service: str='usps') -> str:
        if False:
            while True:
                i = 10
        'Generate random tracking number.\n\n        Supported services: USPS, FedEx and UPS.\n\n        :param str service: Post service.\n        :return: Tracking number.\n        '
        service = service.lower()
        if service not in ('usps', 'fedex', 'ups'):
            raise ValueError('Unsupported post service')
        services = {'usps': ('#### #### #### #### ####', '@@ ### ### ### US'), 'fedex': ('#### #### ####', '#### #### #### ###'), 'ups': ('1Z@####@##########',)}
        mask = self.random.choice(services[service])
        return self.random.custom_code(mask=mask)

    def ssn(self) -> str:
        if False:
            print('Hello World!')
        'Generate a random, but valid SSN.\n\n        :returns: SSN.\n\n        :Example:\n            569-66-5801\n        '
        area = self.random.randint(1, 899)
        if area == 666:
            area = 665
        return '{:03}-{:02}-{:04}'.format(area, self.random.randint(1, 99), self.random.randint(1, 9999))