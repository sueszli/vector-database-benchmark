import json
from django.contrib.gis.db.models.fields import BaseSpatialField
from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.db.models.lookups import DistanceLookupBase, GISLookup
from django.contrib.gis.gdal import GDALRaster
from django.contrib.gis.geos import GEOSGeometry
from django.contrib.gis.measure import D
from django.contrib.gis.shortcuts import numpy
from django.db import connection
from django.db.models import F, Func, Q
from django.test import TransactionTestCase, skipUnlessDBFeature
from django.test.utils import CaptureQueriesContext
from ..data.rasters.textrasters import JSON_RASTER
from .models import RasterModel, RasterRelatedModel

@skipUnlessDBFeature('supports_raster')
class RasterFieldTest(TransactionTestCase):
    available_apps = ['gis_tests.rasterapp']

    def setUp(self):
        if False:
            return 10
        rast = GDALRaster({'srid': 4326, 'origin': [0, 0], 'scale': [-1, 1], 'skew': [0, 0], 'width': 5, 'height': 5, 'nr_of_bands': 2, 'bands': [{'data': range(25)}, {'data': range(25, 50)}]})
        model_instance = RasterModel.objects.create(rast=rast, rastprojected=rast, geom='POINT (-95.37040 29.70486)')
        RasterRelatedModel.objects.create(rastermodel=model_instance)

    def test_field_null_value(self):
        if False:
            return 10
        '\n        Test creating a model where the RasterField has a null value.\n        '
        r = RasterModel.objects.create(rast=None)
        r.refresh_from_db()
        self.assertIsNone(r.rast)

    def test_access_band_data_directly_from_queryset(self):
        if False:
            return 10
        RasterModel.objects.create(rast=JSON_RASTER)
        qs = RasterModel.objects.all()
        qs[0].rast.bands[0].data()

    def test_deserialize_with_pixeltype_flags(self):
        if False:
            return 10
        no_data = 3
        rast = GDALRaster({'srid': 4326, 'origin': [0, 0], 'scale': [-1, 1], 'skew': [0, 0], 'width': 1, 'height': 1, 'nr_of_bands': 1, 'bands': [{'data': [no_data], 'nodata_value': no_data}]})
        r = RasterModel.objects.create(rast=rast)
        RasterModel.objects.filter(pk=r.pk).update(rast=Func(F('rast'), function='ST_SetBandIsNoData'))
        r.refresh_from_db()
        band = r.rast.bands[0].data()
        if numpy:
            band = band.flatten().tolist()
        self.assertEqual(band, [no_data])
        self.assertEqual(r.rast.bands[0].nodata_value, no_data)

    def test_model_creation(self):
        if False:
            return 10
        '\n        Test RasterField through a test model.\n        '
        r = RasterModel.objects.create(rast=JSON_RASTER)
        r.refresh_from_db()
        self.assertEqual((5, 5), (r.rast.width, r.rast.height))
        self.assertEqual([0.0, -1.0, 0.0, 0.0, 0.0, 1.0], r.rast.geotransform)
        self.assertIsNone(r.rast.bands[0].nodata_value)
        self.assertEqual(r.rast.srs.srid, 4326)
        band = r.rast.bands[0].data()
        if numpy:
            band = band.flatten().tolist()
        self.assertEqual([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0], band)

    def test_implicit_raster_transformation(self):
        if False:
            print('Hello World!')
        '\n        Test automatic transformation of rasters with srid different from the\n        field srid.\n        '
        rast = json.loads(JSON_RASTER)
        rast['srid'] = 3086
        r = RasterModel.objects.create(rast=rast)
        r.refresh_from_db()
        self.assertEqual(r.rast.srs.srid, 4326)
        expected = [-87.9298551266551, 9.459646421449934e-06, 0.0, 23.94249275457565, 0.0, -9.459646421449934e-06]
        for (val, exp) in zip(r.rast.geotransform, expected):
            self.assertAlmostEqual(exp, val)

    def test_verbose_name_arg(self):
        if False:
            i = 10
            return i + 15
        '\n        RasterField should accept a positional verbose name argument.\n        '
        self.assertEqual(RasterModel._meta.get_field('rast').verbose_name, 'A Verbose Raster Name')

    def test_all_gis_lookups_with_rasters(self):
        if False:
            print('Hello World!')
        "\n        Evaluate all possible lookups for all input combinations (i.e.\n        raster-raster, raster-geom, geom-raster) and for projected and\n        unprojected coordinate systems. This test just checks that the lookup\n        can be called, but doesn't check if the result makes logical sense.\n        "
        from django.contrib.gis.db.backends.postgis.operations import PostGISOperations
        rast = GDALRaster(json.loads(JSON_RASTER))
        stx_pnt = GEOSGeometry('POINT (-95.370401017314293 29.704867409475465)', 4326)
        stx_pnt.transform(3086)
        lookups = [(name, lookup) for (name, lookup) in BaseSpatialField.get_lookups().items() if issubclass(lookup, GISLookup)]
        self.assertNotEqual(lookups, [], 'No lookups found')
        for (name, lookup) in lookups:
            combo_keys = [field + name for field in ['rast__', 'rast__', 'rastprojected__0__', 'rast__', 'rastprojected__', 'geom__', 'rast__']]
            if issubclass(lookup, DistanceLookupBase):
                combo_values = [(rast, 50, 'spheroid'), (rast, 0, 50, 'spheroid'), (rast, 0, D(km=1)), (stx_pnt, 0, 500), (stx_pnt, D(km=1000)), (rast, 500), (json.loads(JSON_RASTER), 500)]
            elif name == 'relate':
                combo_values = [(rast, 'T*T***FF*'), (rast, 0, 'T*T***FF*'), (rast, 0, 'T*T***FF*'), (stx_pnt, 0, 'T*T***FF*'), (stx_pnt, 'T*T***FF*'), (rast, 'T*T***FF*'), (json.loads(JSON_RASTER), 'T*T***FF*')]
            elif name == 'isvalid':
                continue
            elif PostGISOperations.gis_operators[name].func:
                combo_values = [rast, (rast, 0), (rast, 0), (stx_pnt, 0), stx_pnt, rast, json.loads(JSON_RASTER)]
            else:
                combo_keys[2] = 'rastprojected__' + name
                combo_values = [rast, None, rast, stx_pnt, stx_pnt, rast, json.loads(JSON_RASTER)]
            self.assertEqual(len(combo_keys), len(combo_values), 'Number of lookup names and values should be the same')
            combos = [x for x in zip(combo_keys, combo_values) if x[1]]
            self.assertEqual([(n, x) for (n, x) in enumerate(combos) if x in combos[:n]], [], 'There are repeated test lookups')
            combos = [{k: v} for (k, v) in combos]
            for combo in combos:
                qs = RasterModel.objects.filter(**combo)
                self.assertIn(qs.count(), [0, 1])
            qs = RasterModel.objects.filter(Q(**combos[0]) & Q(**combos[1]))
            self.assertIn(qs.count(), [0, 1])

    def test_dwithin_gis_lookup_output_with_rasters(self):
        if False:
            print('Hello World!')
        '\n        Check the logical functionality of the dwithin lookup for different\n        input parameters.\n        '
        rast = GDALRaster(json.loads(JSON_RASTER))
        stx_pnt = GEOSGeometry('POINT (-95.370401017314293 29.704867409475465)', 4326)
        stx_pnt.transform(3086)
        qs = RasterModel.objects.filter(rastprojected__dwithin=(rast, D(km=1)))
        self.assertEqual(qs.count(), 1)
        qs = RasterModel.objects.filter(rastprojected__dwithin=(json.loads(JSON_RASTER), D(km=1)))
        self.assertEqual(qs.count(), 1)
        qs = RasterModel.objects.filter(rastprojected__dwithin=(JSON_RASTER, D(km=1)))
        self.assertEqual(qs.count(), 1)
        qs = RasterModel.objects.filter(rast__dwithin=(rast, 40))
        self.assertEqual(qs.count(), 1)
        qs = RasterModel.objects.filter(rast__1__dwithin=(rast, 1, 40))
        self.assertEqual(qs.count(), 1)
        qs = RasterModel.objects.filter(rast__1__dwithin=(rast, 40))
        self.assertEqual(qs.count(), 1)
        qs = RasterModel.objects.filter(rast__dwithin=(rast, 1, 40))
        self.assertEqual(qs.count(), 1)
        qs = RasterModel.objects.filter(rast__dwithin=(stx_pnt, 500))
        self.assertEqual(qs.count(), 1)
        qs = RasterModel.objects.filter(rastprojected__dwithin=(stx_pnt, D(km=10000)))
        self.assertEqual(qs.count(), 1)
        qs = RasterModel.objects.filter(rast__dwithin=(stx_pnt, 5))
        self.assertEqual(qs.count(), 0)
        qs = RasterModel.objects.filter(rastprojected__dwithin=(stx_pnt, D(km=100)))
        self.assertEqual(qs.count(), 0)
        qs = RasterModel.objects.filter(geom__dwithin=(rast, 500))
        self.assertEqual(qs.count(), 1)
        qs = RasterRelatedModel.objects.filter(rastermodel__rast__dwithin=(rast, 40))
        self.assertEqual(qs.count(), 1)
        qs = RasterRelatedModel.objects.filter(rastermodel__rast__1__dwithin=(rast, 40))
        self.assertEqual(qs.count(), 1)
        qs = RasterModel.objects.filter(Q(rast__dwithin=(rast, 40)) & Q(rastprojected__dwithin=(stx_pnt, D(km=10000))))
        self.assertEqual(qs.count(), 1)
        qs = RasterModel.objects.filter(rastprojected__bbcontains=rast)
        self.assertEqual(qs.count(), 1)

    def test_lookup_input_tuple_too_long(self):
        if False:
            return 10
        rast = GDALRaster(json.loads(JSON_RASTER))
        msg = 'Tuple too long for lookup bbcontains.'
        with self.assertRaisesMessage(ValueError, msg):
            RasterModel.objects.filter(rast__bbcontains=(rast, 1, 2))

    def test_lookup_input_band_not_allowed(self):
        if False:
            print('Hello World!')
        rast = GDALRaster(json.loads(JSON_RASTER))
        qs = RasterModel.objects.filter(rast__bbcontains=(rast, 1))
        msg = 'Band indices are not allowed for this operator, it works on bbox only.'
        with self.assertRaisesMessage(ValueError, msg):
            qs.count()

    def test_isvalid_lookup_with_raster_error(self):
        if False:
            return 10
        qs = RasterModel.objects.filter(rast__isvalid=True)
        msg = 'IsValid function requires a GeometryField in position 1, got RasterField.'
        with self.assertRaisesMessage(TypeError, msg):
            qs.count()

    def test_result_of_gis_lookup_with_rasters(self):
        if False:
            while True:
                i = 10
        qs = RasterModel.objects.filter(rast__contains=GEOSGeometry('POINT (-0.5 0.5)', 4326))
        self.assertEqual(qs.count(), 1)
        qs = RasterModel.objects.filter(rast__contains=GEOSGeometry('POINT (0.5 0.5)', 4326))
        self.assertEqual(qs.count(), 0)
        qs = RasterModel.objects.filter(rast__contains_properly=GEOSGeometry('POINT (0 0)', 4326))
        self.assertEqual(qs.count(), 0)
        qs = RasterModel.objects.filter(rast__left=GEOSGeometry('POINT (1 0)', 4326))
        self.assertEqual(qs.count(), 1)

    def test_lookup_with_raster_bbox(self):
        if False:
            for i in range(10):
                print('nop')
        rast = GDALRaster(json.loads(JSON_RASTER))
        rast.origin.y = 2
        qs = RasterModel.objects.filter(rast__strictly_below=rast)
        self.assertEqual(qs.count(), 0)
        rast.origin.y = 6
        qs = RasterModel.objects.filter(rast__strictly_below=rast)
        self.assertEqual(qs.count(), 1)

    def test_lookup_with_polygonized_raster(self):
        if False:
            print('Hello World!')
        rast = GDALRaster(json.loads(JSON_RASTER))
        rast.origin.x = -95.3704 + 1
        rast.origin.y = 29.70486
        qs = RasterModel.objects.filter(geom__intersects=rast)
        self.assertEqual(qs.count(), 1)
        rast.bands[0].data(data=[0, 0, 0, 1, 1], shape=(5, 1))
        rast.bands[0].nodata_value = 0
        qs = RasterModel.objects.filter(geom__intersects=rast)
        self.assertEqual(qs.count(), 0)

    def test_lookup_value_error(self):
        if False:
            while True:
                i = 10
        obj = {}
        msg = "Couldn't create spatial object from lookup value '%s'." % obj
        with self.assertRaisesMessage(ValueError, msg):
            RasterModel.objects.filter(geom__intersects=obj)
        obj = '00000'
        msg = "Couldn't create spatial object from lookup value '%s'." % obj
        with self.assertRaisesMessage(ValueError, msg):
            RasterModel.objects.filter(geom__intersects=obj)

    def test_db_function_errors(self):
        if False:
            return 10
        '\n        Errors are raised when using DB functions with raster content.\n        '
        point = GEOSGeometry('SRID=3086;POINT (-697024.9213808845 683729.1705516104)')
        rast = GDALRaster(json.loads(JSON_RASTER))
        msg = 'Distance function requires a geometric argument in position 2.'
        with self.assertRaisesMessage(TypeError, msg):
            RasterModel.objects.annotate(distance_from_point=Distance('geom', rast))
        with self.assertRaisesMessage(TypeError, msg):
            RasterModel.objects.annotate(distance_from_point=Distance('rastprojected', rast))
        msg = 'Distance function requires a GeometryField in position 1, got RasterField.'
        with self.assertRaisesMessage(TypeError, msg):
            RasterModel.objects.annotate(distance_from_point=Distance('rastprojected', point)).count()

    def test_lhs_with_index_rhs_without_index(self):
        if False:
            print('Hello World!')
        with CaptureQueriesContext(connection) as queries:
            RasterModel.objects.filter(rast__0__contains=json.loads(JSON_RASTER)).exists()
        self.assertRegex(queries[-1]['sql'], 'WHERE ST_Contains\\([^)]*, 1, [^)]*, 1\\)')