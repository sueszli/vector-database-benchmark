import os
import shutil
import struct
import tempfile
import zipfile
from pathlib import Path
from unittest import mock
from django.contrib.gis.gdal import GDAL_VERSION, GDALRaster, SpatialReference
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.raster.band import GDALBand
from django.contrib.gis.shortcuts import numpy
from django.core.files.temp import NamedTemporaryFile
from django.test import SimpleTestCase
from ..data.rasters.textrasters import JSON_RASTER

class GDALRasterTests(SimpleTestCase):
    """
    Test a GDALRaster instance created from a file (GeoTiff).
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.rs_path = os.path.join(os.path.dirname(__file__), '../data/rasters/raster.tif')
        self.rs = GDALRaster(self.rs_path)

    def test_gdalraster_input_as_path(self):
        if False:
            while True:
                i = 10
        rs_path = Path(__file__).parent.parent / 'data' / 'rasters' / 'raster.tif'
        rs = GDALRaster(rs_path)
        self.assertEqual(str(rs_path), rs.name)

    def test_rs_name_repr(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.rs_path, self.rs.name)
        self.assertRegex(repr(self.rs), '<Raster object at 0x\\w+>')

    def test_rs_driver(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.rs.driver.name, 'GTiff')

    def test_rs_size(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.rs.width, 163)
        self.assertEqual(self.rs.height, 174)

    def test_rs_srs(self):
        if False:
            return 10
        self.assertEqual(self.rs.srs.srid, 3086)
        self.assertEqual(self.rs.srs.units, (1.0, 'metre'))

    def test_rs_srid(self):
        if False:
            i = 10
            return i + 15
        rast = GDALRaster({'width': 16, 'height': 16, 'srid': 4326})
        self.assertEqual(rast.srid, 4326)
        rast.srid = 3086
        self.assertEqual(rast.srid, 3086)

    def test_geotransform_and_friends(self):
        if False:
            return 10
        self.assertEqual(self.rs.geotransform, [511700.4680706557, 100.0, 0.0, 435103.3771231986, 0.0, -100.0])
        self.assertEqual(self.rs.origin, [511700.4680706557, 435103.3771231986])
        self.assertEqual(self.rs.origin.x, 511700.4680706557)
        self.assertEqual(self.rs.origin.y, 435103.3771231986)
        self.assertEqual(self.rs.scale, [100.0, -100.0])
        self.assertEqual(self.rs.scale.x, 100.0)
        self.assertEqual(self.rs.scale.y, -100.0)
        self.assertEqual(self.rs.skew, [0, 0])
        self.assertEqual(self.rs.skew.x, 0)
        self.assertEqual(self.rs.skew.y, 0)
        rsmem = GDALRaster(JSON_RASTER)
        rsmem.geotransform = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(rsmem.geotransform, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        rsmem.geotransform = range(6)
        self.assertEqual(rsmem.geotransform, [float(x) for x in range(6)])
        self.assertEqual(rsmem.origin, [0, 3])
        self.assertEqual(rsmem.origin.x, 0)
        self.assertEqual(rsmem.origin.y, 3)
        self.assertEqual(rsmem.scale, [1, 5])
        self.assertEqual(rsmem.scale.x, 1)
        self.assertEqual(rsmem.scale.y, 5)
        self.assertEqual(rsmem.skew, [2, 4])
        self.assertEqual(rsmem.skew.x, 2)
        self.assertEqual(rsmem.skew.y, 4)
        self.assertEqual(rsmem.width, 5)
        self.assertEqual(rsmem.height, 5)

    def test_geotransform_bad_inputs(self):
        if False:
            return 10
        rsmem = GDALRaster(JSON_RASTER)
        error_geotransforms = [[1, 2], [1, 2, 3, 4, 5, 'foo'], [1, 2, 3, 4, 5, 6, 'foo']]
        msg = 'Geotransform must consist of 6 numeric values.'
        for geotransform in error_geotransforms:
            with self.subTest(i=geotransform), self.assertRaisesMessage(ValueError, msg):
                rsmem.geotransform = geotransform

    def test_rs_extent(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.rs.extent, (511700.4680706557, 417703.3771231986, 528000.4680706557, 435103.3771231986))

    def test_rs_bands(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(self.rs.bands), 1)
        self.assertIsInstance(self.rs.bands[0], GDALBand)

    def test_memory_based_raster_creation(self):
        if False:
            i = 10
            return i + 15
        rast = GDALRaster({'datatype': 1, 'width': 16, 'height': 16, 'srid': 4326, 'bands': [{'data': range(256), 'nodata_value': 255}]})
        result = rast.bands[0].data()
        if numpy:
            result = result.flatten().tolist()
        self.assertEqual(result, list(range(256)))

    def test_file_based_raster_creation(self):
        if False:
            print('Hello World!')
        rstfile = NamedTemporaryFile(suffix='.tif')
        GDALRaster({'datatype': self.rs.bands[0].datatype(), 'driver': 'tif', 'name': rstfile.name, 'width': 163, 'height': 174, 'nr_of_bands': 1, 'srid': self.rs.srs.wkt, 'origin': (self.rs.origin.x, self.rs.origin.y), 'scale': (self.rs.scale.x, self.rs.scale.y), 'skew': (self.rs.skew.x, self.rs.skew.y), 'bands': [{'data': self.rs.bands[0].data(), 'nodata_value': self.rs.bands[0].nodata_value}]})
        restored_raster = GDALRaster(rstfile.name)
        self.assertEqual(restored_raster.srs.wkt.replace('TOWGS84[0,0,0,0,0,0,0],', ''), self.rs.srs.wkt.replace('TOWGS84[0,0,0,0,0,0,0],', ''))
        self.assertEqual(restored_raster.geotransform, self.rs.geotransform)
        if numpy:
            numpy.testing.assert_equal(restored_raster.bands[0].data(), self.rs.bands[0].data())
        else:
            self.assertEqual(restored_raster.bands[0].data(), self.rs.bands[0].data())

    def test_nonexistent_file(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Unable to read raster source input "nonexistent.tif".'
        with self.assertRaisesMessage(GDALException, msg):
            GDALRaster('nonexistent.tif')

    def test_vsi_raster_creation(self):
        if False:
            print('Hello World!')
        with open(self.rs_path, 'rb') as dat:
            vsimem = GDALRaster(dat.read())
        result = vsimem.bands[0].data()
        target = self.rs.bands[0].data()
        if numpy:
            result = result.flatten().tolist()
            target = target.flatten().tolist()
        self.assertEqual(result, target)

    def test_vsi_raster_deletion(self):
        if False:
            print('Hello World!')
        path = '/vsimem/raster.tif'
        vsimem = GDALRaster({'name': path, 'driver': 'tif', 'width': 4, 'height': 4, 'srid': 4326, 'bands': [{'data': range(16)}]})
        rst = GDALRaster(path)
        self.assertEqual(rst.width, 4)
        del vsimem
        del rst
        msg = 'Could not open the datasource at "/vsimem/raster.tif"'
        with self.assertRaisesMessage(GDALException, msg):
            GDALRaster(path)

    def test_vsi_invalid_buffer_error(self):
        if False:
            i = 10
            return i + 15
        msg = 'Failed creating VSI raster from the input buffer.'
        with self.assertRaisesMessage(GDALException, msg):
            GDALRaster(b'not-a-raster-buffer')

    def test_vsi_buffer_property(self):
        if False:
            for i in range(10):
                print('nop')
        rast = GDALRaster({'name': '/vsimem/raster.tif', 'driver': 'tif', 'width': 4, 'height': 4, 'srid': 4326, 'bands': [{'data': range(16)}]})
        result = GDALRaster(rast.vsi_buffer).bands[0].data()
        if numpy:
            result = result.flatten().tolist()
        self.assertEqual(result, list(range(16)))
        self.assertIsNone(self.rs.vsi_buffer)

    def test_vsi_vsizip_filesystem(self):
        if False:
            i = 10
            return i + 15
        rst_zipfile = NamedTemporaryFile(suffix='.zip')
        with zipfile.ZipFile(rst_zipfile, mode='w') as zf:
            zf.write(self.rs_path, 'raster.tif')
        rst_path = '/vsizip/' + os.path.join(rst_zipfile.name, 'raster.tif')
        rst = GDALRaster(rst_path)
        self.assertEqual(rst.driver.name, self.rs.driver.name)
        self.assertEqual(rst.name, rst_path)
        self.assertIs(rst.is_vsi_based, True)
        self.assertIsNone(rst.vsi_buffer)

    def test_offset_size_and_shape_on_raster_creation(self):
        if False:
            i = 10
            return i + 15
        rast = GDALRaster({'datatype': 1, 'width': 4, 'height': 4, 'srid': 4326, 'bands': [{'data': (1,), 'offset': (1, 1), 'size': (2, 2), 'shape': (1, 1), 'nodata_value': 2}]})
        result = rast.bands[0].data()
        if numpy:
            result = result.flatten().tolist()
        self.assertEqual(result, [2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2])

    def test_set_nodata_value_on_raster_creation(self):
        if False:
            for i in range(10):
                print('nop')
        rast = GDALRaster({'datatype': 1, 'width': 2, 'height': 2, 'srid': 4326, 'bands': [{'nodata_value': 23}]})
        result = rast.bands[0].data()
        if numpy:
            result = result.flatten().tolist()
        self.assertEqual(result, [23] * 4)

    def test_set_nodata_none_on_raster_creation(self):
        if False:
            i = 10
            return i + 15
        rast = GDALRaster({'datatype': 1, 'width': 2, 'height': 2, 'srid': 4326, 'bands': [{'nodata_value': None}]})
        result = rast.bands[0].data()
        if numpy:
            result = result.flatten().tolist()
        self.assertEqual(result, [0] * 4)

    def test_raster_metadata_property(self):
        if False:
            return 10
        data = self.rs.metadata
        self.assertEqual(data['DEFAULT'], {'AREA_OR_POINT': 'Area'})
        self.assertEqual(data['IMAGE_STRUCTURE'], {'INTERLEAVE': 'BAND'})
        source = GDALRaster({'datatype': 1, 'width': 2, 'height': 2, 'srid': 4326, 'bands': [{'data': range(4), 'nodata_value': 99}]})
        metadata = {'DEFAULT': {'OWNER': 'Django', 'VERSION': '1.0', 'AREA_OR_POINT': 'Point'}}
        source.metadata = metadata
        source.bands[0].metadata = metadata
        self.assertEqual(source.metadata['DEFAULT'], metadata['DEFAULT'])
        self.assertEqual(source.bands[0].metadata['DEFAULT'], metadata['DEFAULT'])
        metadata = {'DEFAULT': {'VERSION': '2.0'}}
        source.metadata = metadata
        self.assertEqual(source.metadata['DEFAULT']['VERSION'], '2.0')
        metadata = {'DEFAULT': {'OWNER': None}}
        source.metadata = metadata
        self.assertNotIn('OWNER', source.metadata['DEFAULT'])

    def test_raster_info_accessor(self):
        if False:
            for i in range(10):
                print('nop')
        infos = self.rs.info
        info_lines = [line.strip() for line in infos.split('\n') if line.strip() != '']
        for line in ['Driver: GTiff/GeoTIFF', 'Files: {}'.format(self.rs_path), 'Size is 163, 174', 'Origin = (511700.468070655711927,435103.377123198588379)', 'Pixel Size = (100.000000000000000,-100.000000000000000)', 'Metadata:', 'AREA_OR_POINT=Area', 'Image Structure Metadata:', 'INTERLEAVE=BAND', 'Band 1 Block=163x50 Type=Byte, ColorInterp=Gray', 'NoData Value=15']:
            self.assertIn(line, info_lines)
        for line in ['Upper Left  \\(  511700.468,  435103.377\\) \\( 82d51\\\'46.1\\d"W, 27d55\\\' 1.5\\d"N\\)', 'Lower Left  \\(  511700.468,  417703.377\\) \\( 82d51\\\'52.0\\d"W, 27d45\\\'37.5\\d"N\\)', 'Upper Right \\(  528000.468,  435103.377\\) \\( 82d41\\\'48.8\\d"W, 27d54\\\'56.3\\d"N\\)', 'Lower Right \\(  528000.468,  417703.377\\) \\( 82d41\\\'55.5\\d"W, 27d45\\\'32.2\\d"N\\)', 'Center      \\(  519850.468,  426403.377\\) \\( 82d46\\\'50.6\\d"W, 27d50\\\'16.9\\d"N\\)']:
            self.assertRegex(infos, line)
        self.assertIn('NAD83 / Florida GDL Albers', infos)

    def test_compressed_file_based_raster_creation(self):
        if False:
            print('Hello World!')
        rstfile = NamedTemporaryFile(suffix='.tif')
        compressed = self.rs.warp({'papsz_options': {'compress': 'packbits'}, 'name': rstfile.name})
        self.assertLess(os.path.getsize(compressed.name), os.path.getsize(self.rs.name))
        papsz_options = {'compress': 'packbits', 'blockxsize': 23, 'blockysize': 23}
        if GDAL_VERSION < (3, 7):
            datatype = 1
            papsz_options['pixeltype'] = 'signedbyte'
        else:
            datatype = 14
        compressed = GDALRaster({'datatype': datatype, 'driver': 'tif', 'name': rstfile.name, 'width': 40, 'height': 40, 'srid': 3086, 'origin': (500000, 400000), 'scale': (100, -100), 'skew': (0, 0), 'bands': [{'data': range(40 ^ 2), 'nodata_value': 255}], 'papsz_options': papsz_options})
        compressed = GDALRaster(compressed.name)
        self.assertEqual(compressed.metadata['IMAGE_STRUCTURE']['COMPRESSION'], 'PACKBITS')
        self.assertEqual(compressed.bands[0].datatype(), datatype)
        if GDAL_VERSION < (3, 7):
            self.assertEqual(compressed.bands[0].metadata['IMAGE_STRUCTURE']['PIXELTYPE'], 'SIGNEDBYTE')
        self.assertIn('Block=40x23', compressed.info)

    def test_raster_warp(self):
        if False:
            print('Hello World!')
        source = GDALRaster({'datatype': 1, 'driver': 'MEM', 'name': 'sourceraster', 'width': 4, 'height': 4, 'nr_of_bands': 1, 'srid': 3086, 'origin': (500000, 400000), 'scale': (100, -100), 'skew': (0, 0), 'bands': [{'data': range(16), 'nodata_value': 255}]})
        data = {'scale': [200, -200], 'width': 2, 'height': 2}
        target = source.warp(data)
        self.assertEqual(target.width, data['width'])
        self.assertEqual(target.height, data['height'])
        self.assertEqual(target.scale, data['scale'])
        self.assertEqual(target.bands[0].datatype(), source.bands[0].datatype())
        self.assertEqual(target.name, 'sourceraster_copy.MEM')
        result = target.bands[0].data()
        if numpy:
            result = result.flatten().tolist()
        self.assertEqual(result, [5, 7, 13, 15])
        data = {'name': '/path/to/targetraster.tif', 'datatype': 6}
        target = source.warp(data)
        self.assertEqual(target.bands[0].datatype(), 6)
        self.assertEqual(target.name, '/path/to/targetraster.tif')
        self.assertEqual(target.driver.name, 'MEM')
        result = target.bands[0].data()
        if numpy:
            result = result.flatten().tolist()
        self.assertEqual(result, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

    def test_raster_warp_nodata_zone(self):
        if False:
            print('Hello World!')
        source = GDALRaster({'datatype': 1, 'driver': 'MEM', 'width': 4, 'height': 4, 'srid': 3086, 'origin': (500000, 400000), 'scale': (100, -100), 'skew': (0, 0), 'bands': [{'data': range(16), 'nodata_value': 23}]})
        result = source.warp({'origin': (200000, 200000)}).bands[0].data()
        if numpy:
            result = result.flatten().tolist()
        self.assertEqual(result, [23] * 16)

    def test_raster_clone(self):
        if False:
            for i in range(10):
                print('nop')
        rstfile = NamedTemporaryFile(suffix='.tif')
        tests = [('MEM', '', 23), ('tif', rstfile.name, 99)]
        for (driver, name, nodata_value) in tests:
            with self.subTest(driver=driver):
                source = GDALRaster({'datatype': 1, 'driver': driver, 'name': name, 'width': 4, 'height': 4, 'srid': 3086, 'origin': (500000, 400000), 'scale': (100, -100), 'skew': (0, 0), 'bands': [{'data': range(16), 'nodata_value': nodata_value}]})
                clone = source.clone()
                self.assertNotEqual(clone.name, source.name)
                self.assertEqual(clone._write, source._write)
                self.assertEqual(clone.srs.srid, source.srs.srid)
                self.assertEqual(clone.width, source.width)
                self.assertEqual(clone.height, source.height)
                self.assertEqual(clone.origin, source.origin)
                self.assertEqual(clone.scale, source.scale)
                self.assertEqual(clone.skew, source.skew)
                self.assertIsNot(clone, source)

    def test_raster_transform(self):
        if False:
            i = 10
            return i + 15
        tests = [3086, '3086', SpatialReference(3086)]
        for srs in tests:
            with self.subTest(srs=srs):
                rstfile = NamedTemporaryFile(suffix='.tif')
                ndv = 99
                source = GDALRaster({'datatype': 1, 'driver': 'tif', 'name': rstfile.name, 'width': 5, 'height': 5, 'nr_of_bands': 1, 'srid': 4326, 'origin': (-5, 5), 'scale': (2, -2), 'skew': (0, 0), 'bands': [{'data': range(25), 'nodata_value': ndv}]})
                target = source.transform(srs)
                target = GDALRaster(target.name)
                self.assertEqual(target.srs.srid, 3086)
                self.assertEqual(target.width, 7)
                self.assertEqual(target.height, 7)
                self.assertEqual(target.bands[0].datatype(), source.bands[0].datatype())
                self.assertAlmostEqual(target.origin[0], 9124842.791079799, 3)
                self.assertAlmostEqual(target.origin[1], 1589911.6476407414, 3)
                self.assertAlmostEqual(target.scale[0], 223824.82664250192, 3)
                self.assertAlmostEqual(target.scale[1], -223824.82664250192, 3)
                self.assertEqual(target.skew, [0, 0])
                result = target.bands[0].data()
                if numpy:
                    result = result.flatten().tolist()
                self.assertEqual(result, [ndv, ndv, ndv, ndv, 4, ndv, ndv, ndv, ndv, 2, 3, 9, ndv, ndv, ndv, 1, 2, 8, 13, 19, ndv, 0, 6, 6, 12, 18, 18, 24, ndv, 10, 11, 16, 22, 23, ndv, ndv, ndv, 15, 21, 22, ndv, ndv, ndv, ndv, 20, ndv, ndv, ndv, ndv])

    def test_raster_transform_clone(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch.object(GDALRaster, 'clone') as mocked_clone:
            rstfile = NamedTemporaryFile(suffix='.tif')
            source = GDALRaster({'datatype': 1, 'driver': 'tif', 'name': rstfile.name, 'width': 5, 'height': 5, 'nr_of_bands': 1, 'srid': 4326, 'origin': (-5, 5), 'scale': (2, -2), 'skew': (0, 0), 'bands': [{'data': range(25), 'nodata_value': 99}]})
            source.transform(4326)
            self.assertEqual(mocked_clone.call_count, 1)

    def test_raster_transform_clone_name(self):
        if False:
            print('Hello World!')
        rstfile = NamedTemporaryFile(suffix='.tif')
        source = GDALRaster({'datatype': 1, 'driver': 'tif', 'name': rstfile.name, 'width': 5, 'height': 5, 'nr_of_bands': 1, 'srid': 4326, 'origin': (-5, 5), 'scale': (2, -2), 'skew': (0, 0), 'bands': [{'data': range(25), 'nodata_value': 99}]})
        clone_name = rstfile.name + '_respect_name.GTiff'
        target = source.transform(4326, name=clone_name)
        self.assertEqual(target.name, clone_name)

class GDALBandTests(SimpleTestCase):
    rs_path = os.path.join(os.path.dirname(__file__), '../data/rasters/raster.tif')

    def test_band_data(self):
        if False:
            i = 10
            return i + 15
        rs = GDALRaster(self.rs_path)
        band = rs.bands[0]
        self.assertEqual(band.width, 163)
        self.assertEqual(band.height, 174)
        self.assertEqual(band.description, '')
        self.assertEqual(band.datatype(), 1)
        self.assertEqual(band.datatype(as_string=True), 'GDT_Byte')
        self.assertEqual(band.color_interp(), 1)
        self.assertEqual(band.color_interp(as_string=True), 'GCI_GrayIndex')
        self.assertEqual(band.nodata_value, 15)
        if numpy:
            data = band.data()
            assert_array = numpy.loadtxt(os.path.join(os.path.dirname(__file__), '../data/rasters/raster.numpy.txt'))
            numpy.testing.assert_equal(data, assert_array)
            self.assertEqual(data.shape, (band.height, band.width))

    def test_band_statistics(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmp_dir:
            rs_path = os.path.join(tmp_dir, 'raster.tif')
            shutil.copyfile(self.rs_path, rs_path)
            rs = GDALRaster(rs_path)
            band = rs.bands[0]
            pam_file = rs_path + '.aux.xml'
            (smin, smax, smean, sstd) = band.statistics(approximate=True)
            self.assertEqual(smin, 0)
            self.assertEqual(smax, 9)
            self.assertAlmostEqual(smean, 2.842331288343558)
            self.assertAlmostEqual(sstd, 2.3965567248965356)
            (smin, smax, smean, sstd) = band.statistics(approximate=False, refresh=True)
            self.assertEqual(smin, 0)
            self.assertEqual(smax, 9)
            self.assertAlmostEqual(smean, 2.828326634228898)
            self.assertAlmostEqual(sstd, 2.4260526986669095)
            self.assertEqual(band.min, 0)
            self.assertEqual(band.max, 9)
            self.assertAlmostEqual(band.mean, 2.828326634228898)
            self.assertAlmostEqual(band.std, 2.4260526986669095)
            rs = band = None
            self.assertTrue(os.path.isfile(pam_file))

    def _remove_aux_file(self):
        if False:
            i = 10
            return i + 15
        pam_file = self.rs_path + '.aux.xml'
        if os.path.isfile(pam_file):
            os.remove(pam_file)

    def test_read_mode_error(self):
        if False:
            return 10
        rs = GDALRaster(self.rs_path, write=False)
        band = rs.bands[0]
        self.addCleanup(self._remove_aux_file)
        with self.assertRaises(GDALException):
            setattr(band, 'nodata_value', 10)

    def test_band_data_setters(self):
        if False:
            return 10
        rsmem = GDALRaster({'datatype': 1, 'driver': 'MEM', 'name': 'mem_rst', 'width': 10, 'height': 10, 'nr_of_bands': 1, 'srid': 4326})
        bandmem = rsmem.bands[0]
        bandmem.nodata_value = 99
        self.assertEqual(bandmem.nodata_value, 99)
        bandmem.data(range(100))
        if numpy:
            numpy.testing.assert_equal(bandmem.data(), numpy.arange(100).reshape(10, 10))
        else:
            self.assertEqual(bandmem.data(), list(range(100)))
        block = list(range(100, 104))
        packed_block = struct.pack('<' + 'B B B B', *block)
        bandmem.data(block, (1, 1), (2, 2))
        result = bandmem.data(offset=(1, 1), size=(2, 2))
        if numpy:
            numpy.testing.assert_equal(result, numpy.array(block).reshape(2, 2))
        else:
            self.assertEqual(result, block)
        bandmem.data(packed_block, (1, 1), (2, 2))
        result = bandmem.data(offset=(1, 1), size=(2, 2))
        if numpy:
            numpy.testing.assert_equal(result, numpy.array(block).reshape(2, 2))
        else:
            self.assertEqual(result, block)
        bandmem.data(bytes(packed_block), (1, 1), (2, 2))
        result = bandmem.data(offset=(1, 1), size=(2, 2))
        if numpy:
            numpy.testing.assert_equal(result, numpy.array(block).reshape(2, 2))
        else:
            self.assertEqual(result, block)
        bandmem.data(bytearray(packed_block), (1, 1), (2, 2))
        result = bandmem.data(offset=(1, 1), size=(2, 2))
        if numpy:
            numpy.testing.assert_equal(result, numpy.array(block).reshape(2, 2))
        else:
            self.assertEqual(result, block)
        bandmem.data(memoryview(packed_block), (1, 1), (2, 2))
        result = bandmem.data(offset=(1, 1), size=(2, 2))
        if numpy:
            numpy.testing.assert_equal(result, numpy.array(block).reshape(2, 2))
        else:
            self.assertEqual(result, block)
        if numpy:
            bandmem.data(numpy.array(block, dtype='int8').reshape(2, 2), (1, 1), (2, 2))
            numpy.testing.assert_equal(bandmem.data(offset=(1, 1), size=(2, 2)), numpy.array(block).reshape(2, 2))
        rsmemjson = GDALRaster(JSON_RASTER)
        bandmemjson = rsmemjson.bands[0]
        if numpy:
            numpy.testing.assert_equal(bandmemjson.data(), numpy.array(range(25)).reshape(5, 5))
        else:
            self.assertEqual(bandmemjson.data(), list(range(25)))

    def test_band_statistics_automatic_refresh(self):
        if False:
            while True:
                i = 10
        rsmem = GDALRaster({'srid': 4326, 'width': 2, 'height': 2, 'bands': [{'data': [0] * 4, 'nodata_value': 99}]})
        band = rsmem.bands[0]
        self.assertEqual(band.statistics(), (0, 0, 0, 0))
        band.data([1, 1, 0, 0])
        self.assertEqual(band.statistics(), (0.0, 1.0, 0.5, 0.5))
        band.nodata_value = 0
        self.assertEqual(band.statistics(), (1.0, 1.0, 1.0, 0.0))

    def test_band_statistics_empty_band(self):
        if False:
            return 10
        rsmem = GDALRaster({'srid': 4326, 'width': 1, 'height': 1, 'bands': [{'data': [0], 'nodata_value': 0}]})
        self.assertEqual(rsmem.bands[0].statistics(), (None, None, None, None))

    def test_band_delete_nodata(self):
        if False:
            return 10
        rsmem = GDALRaster({'srid': 4326, 'width': 1, 'height': 1, 'bands': [{'data': [0], 'nodata_value': 1}]})
        rsmem.bands[0].nodata_value = None
        self.assertIsNone(rsmem.bands[0].nodata_value)

    def test_band_data_replication(self):
        if False:
            print('Hello World!')
        band = GDALRaster({'srid': 4326, 'width': 3, 'height': 3, 'bands': [{'data': range(10, 19), 'nodata_value': 0}]}).bands[0]
        combos = (([1], (1, 1), [1] * 9), (range(3), (1, 3), [0, 0, 0, 1, 1, 1, 2, 2, 2]), (range(3), (3, 1), [0, 1, 2, 0, 1, 2, 0, 1, 2]))
        for combo in combos:
            band.data(combo[0], shape=combo[1])
            if numpy:
                numpy.testing.assert_equal(band.data(), numpy.array(combo[2]).reshape(3, 3))
            else:
                self.assertEqual(band.data(), list(combo[2]))