import re
from django.db import connection
from django.test import TestCase, skipUnlessDBFeature
from django.utils.functional import cached_property
test_srs = ({'srid': 4326, 'auth_name': ('EPSG', True), 'auth_srid': 4326, 'srtext': 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84"', 'proj_re': '\\+proj=longlat (\\+datum=WGS84 |\\+towgs84=0,0,0,0,0,0,0 )\\+no_defs ?', 'spheroid': 'WGS 84', 'name': 'WGS 84', 'geographic': True, 'projected': False, 'spatialite': True, 'ellipsoid': (6378137.0, 6356752.3, 298.257223563), 'eprec': (1, 1, 9), 'wkt': re.sub('[\\s+]', '', '\n        GEOGCS["WGS 84",\n    DATUM["WGS_1984",\n        SPHEROID["WGS 84",6378137,298.257223563,\n            AUTHORITY["EPSG","7030"]],\n        AUTHORITY["EPSG","6326"]],\n    PRIMEM["Greenwich",0,\n        AUTHORITY["EPSG","8901"]],\n    UNIT["degree",0.01745329251994328,\n        AUTHORITY["EPSG","9122"]],\n    AUTHORITY["EPSG","4326"]]\n    ')}, {'srid': 32140, 'auth_name': ('EPSG', False), 'auth_srid': 32140, 'srtext': 'PROJCS["NAD83 / Texas South Central",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980"', 'proj_re': '\\+proj=lcc (\\+lat_1=30.28333333333333? |\\+lat_2=28.38333333333333? |\\+lat_0=27.83333333333333? |\\+lon_0=-99 ){4}\\+x_0=600000 \\+y_0=4000000 (\\+ellps=GRS80 )?(\\+datum=NAD83 |\\+towgs84=0,0,0,0,0,0,0 )?\\+units=m \\+no_defs ?', 'spheroid': 'GRS 1980', 'name': 'NAD83 / Texas South Central', 'geographic': False, 'projected': True, 'spatialite': False, 'ellipsoid': (6378137.0, 6356752.31414, 298.257222101), 'eprec': (1, 5, 10)})

@skipUnlessDBFeature('has_spatialrefsys_table')
class SpatialRefSysTest(TestCase):

    @cached_property
    def SpatialRefSys(self):
        if False:
            for i in range(10):
                print('nop')
        return connection.ops.connection.ops.spatial_ref_sys()

    def test_get_units(self):
        if False:
            while True:
                i = 10
        epsg_4326 = next((f for f in test_srs if f['srid'] == 4326))
        (unit, unit_name) = self.SpatialRefSys().get_units(epsg_4326['wkt'])
        self.assertEqual(unit_name, 'degree')
        self.assertAlmostEqual(unit, 0.01745329251994328)

    def test_retrieve(self):
        if False:
            print('Hello World!')
        '\n        Test retrieval of SpatialRefSys model objects.\n        '
        for sd in test_srs:
            srs = self.SpatialRefSys.objects.get(srid=sd['srid'])
            self.assertEqual(sd['srid'], srs.srid)
            (auth_name, oracle_flag) = sd['auth_name']
            if not connection.ops.oracle or oracle_flag:
                self.assertIs(srs.auth_name.upper().startswith(auth_name), True)
            self.assertEqual(sd['auth_srid'], srs.auth_srid)
            if not connection.ops.oracle:
                self.assertTrue(srs.wkt.startswith(sd['srtext']))
                self.assertRegex(srs.proj4text, sd['proj_re'])

    def test_osr(self):
        if False:
            print('Hello World!')
        '\n        Test getting OSR objects from SpatialRefSys model objects.\n        '
        for sd in test_srs:
            sr = self.SpatialRefSys.objects.get(srid=sd['srid'])
            self.assertTrue(sr.spheroid.startswith(sd['spheroid']))
            self.assertEqual(sd['geographic'], sr.geographic)
            self.assertEqual(sd['projected'], sr.projected)
            self.assertIs(sr.name.startswith(sd['name']), True)
            if not connection.ops.oracle:
                srs = sr.srs
                self.assertRegex(srs.proj, sd['proj_re'])
                self.assertTrue(srs.wkt.startswith(sd['srtext']))

    def test_ellipsoid(self):
        if False:
            print('Hello World!')
        '\n        Test the ellipsoid property.\n        '
        for sd in test_srs:
            ellps1 = sd['ellipsoid']
            prec = sd['eprec']
            srs = self.SpatialRefSys.objects.get(srid=sd['srid'])
            ellps2 = srs.ellipsoid
            for i in range(3):
                self.assertAlmostEqual(ellps1[i], ellps2[i], prec[i])

    @skipUnlessDBFeature('supports_add_srs_entry')
    def test_add_entry(self):
        if False:
            while True:
                i = 10
        '\n        Test adding a new entry in the SpatialRefSys model using the\n        add_srs_entry utility.\n        '
        from django.contrib.gis.utils import add_srs_entry
        add_srs_entry(3857)
        self.assertTrue(self.SpatialRefSys.objects.filter(srid=3857).exists())
        srs = self.SpatialRefSys.objects.get(srid=3857)
        self.assertTrue(self.SpatialRefSys.get_spheroid(srs.wkt).startswith('SPHEROID['))