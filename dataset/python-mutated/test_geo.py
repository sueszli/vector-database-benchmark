import re
import unittest
from decimal import Decimal
from faker import Faker
from faker.providers.geo.pt_PT import Provider as PtPtProvider

class TestGlobal(unittest.TestCase):
    """Tests geographic locations regardless of locale"""

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fake = Faker()
        Faker.seed(0)

    def test_local_latlng(self):
        if False:
            return 10
        loc = self.fake.local_latlng(country_code='US')
        assert isinstance(loc, tuple)
        assert len(loc) == 5
        assert Decimal(loc[0])
        assert Decimal(loc[1])
        loc_short = self.fake.local_latlng(country_code='US', coords_only=True)
        assert len(loc_short) == 2
        assert Decimal(loc_short[0])
        assert Decimal(loc_short[1])

class TestEnUS(unittest.TestCase):
    """Tests geographic locations in the en_US locale"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fake = Faker('en_US')
        Faker.seed(0)

    def test_latitude(self):
        if False:
            for i in range(10):
                print('nop')
        lat = self.fake.latitude()
        assert isinstance(lat, Decimal)

    def test_longitude(self):
        if False:
            return 10
        long = self.fake.longitude()
        assert isinstance(long, Decimal)

    def test_latlng(self):
        if False:
            i = 10
            return i + 15
        loc = self.fake.latlng()
        assert isinstance(loc, tuple)
        assert len(loc) == 2
        assert isinstance(loc[0], Decimal)
        assert isinstance(loc[1], Decimal)

    def test_coordinate(self):
        if False:
            print('Hello World!')
        loc = self.fake.coordinate()
        assert isinstance(loc, Decimal)

    def test_coordinate_centered(self):
        if False:
            return 10
        loc = self.fake.coordinate(center=23)
        assert round(loc) == 23

    def test_coordinate_rounded(self):
        if False:
            print('Hello World!')
        loc = self.fake.coordinate(center=23, radius=3)
        assert 20 <= round(loc) <= 26

    def test_location_on_land(self):
        if False:
            return 10
        loc = self.fake.location_on_land()
        assert isinstance(loc, tuple)
        assert len(loc) == 5
        assert Decimal(loc[0])
        assert Decimal(loc[1])
        assert isinstance(loc[2], str)
        assert isinstance(loc[3], str)
        assert len(loc[3]) == 2
        assert isinstance(loc[4], str)

    def test_location_on_land_coords_only(self):
        if False:
            i = 10
            return i + 15
        loc = self.fake.location_on_land(coords_only=True)
        assert isinstance(loc, tuple)
        assert len(loc) == 2
        assert Decimal(loc[0])
        assert Decimal(loc[1])

class TestCsCz(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.fake = Faker('cs_CZ')
        Faker.seed(0)

    def test_location_on_land(self):
        if False:
            i = 10
            return i + 15
        loc = self.fake.location_on_land()
        assert isinstance(loc, tuple)
        assert len(loc) == 5
        assert Decimal(loc[0])
        assert Decimal(loc[1])
        assert isinstance(loc[2], str)
        assert isinstance(loc[3], str)
        assert len(loc[3]) == 2
        assert isinstance(loc[4], str)

class TestDeAT(unittest.TestCase):
    """Tests in addresses in the de_AT locale"""

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fake = Faker('de_AT')
        Faker.seed(0)

    def test_local_latitude(self):
        if False:
            while True:
                i = 10
        local_latitude = self.fake.local_latitude()
        assert re.match('4[6-8]\\.\\d+', str(local_latitude))

    def test_local_longitude(self):
        if False:
            print('Hello World!')
        local_longitude = self.fake.local_longitude()
        assert re.match('1[1-5]\\.\\d+', str(local_longitude))

class TestPtPT(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fake = Faker('pt_PT')
        Faker.seed(0)

    def test_nationality(self):
        if False:
            print('Hello World!')
        nationality = self.fake.nationality()
        assert isinstance(nationality, str)
        assert nationality in PtPtProvider.nationalities

class TestTrTr(TestEnUS):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fake = Faker('tr_TR')
        Faker.seed(0)

class TestEnIe(TestEnUS):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fake = Faker('en_IE')
        Faker.seed(0)