"""
 The LayerMapping class provides a way to map the contents of OGR
 vector files (e.g. SHP files) to Geographic-enabled Django models.

 For more information, please consult the GeoDjango documentation:
   https://docs.djangoproject.com/en/dev/ref/contrib/gis/layermapping/
"""
import sys
from decimal import Decimal
from decimal import InvalidOperation as DecimalInvalidOperation
from pathlib import Path
from django.contrib.gis.db.models import GeometryField
from django.contrib.gis.gdal import CoordTransform, DataSource, GDALException, OGRGeometry, OGRGeomType, SpatialReference
from django.contrib.gis.gdal.field import OFTDate, OFTDateTime, OFTInteger, OFTInteger64, OFTReal, OFTString, OFTTime
from django.core.exceptions import FieldDoesNotExist, ObjectDoesNotExist
from django.db import connections, models, router, transaction
from django.utils.encoding import force_str

class LayerMapError(Exception):
    pass

class InvalidString(LayerMapError):
    pass

class InvalidDecimal(LayerMapError):
    pass

class InvalidInteger(LayerMapError):
    pass

class MissingForeignKey(LayerMapError):
    pass

class LayerMapping:
    """A class that maps OGR Layers to GeoDjango Models."""
    MULTI_TYPES = {1: OGRGeomType('MultiPoint'), 2: OGRGeomType('MultiLineString'), 3: OGRGeomType('MultiPolygon'), OGRGeomType('Point25D').num: OGRGeomType('MultiPoint25D'), OGRGeomType('LineString25D').num: OGRGeomType('MultiLineString25D'), OGRGeomType('Polygon25D').num: OGRGeomType('MultiPolygon25D')}
    FIELD_TYPES = {models.AutoField: OFTInteger, models.BigAutoField: OFTInteger64, models.SmallAutoField: OFTInteger, models.BooleanField: (OFTInteger, OFTReal, OFTString), models.IntegerField: (OFTInteger, OFTReal, OFTString), models.FloatField: (OFTInteger, OFTReal), models.DateField: OFTDate, models.DateTimeField: OFTDateTime, models.EmailField: OFTString, models.TimeField: OFTTime, models.DecimalField: (OFTInteger, OFTReal), models.CharField: OFTString, models.SlugField: OFTString, models.TextField: OFTString, models.URLField: OFTString, models.UUIDField: OFTString, models.BigIntegerField: (OFTInteger, OFTReal, OFTString), models.SmallIntegerField: (OFTInteger, OFTReal, OFTString), models.PositiveBigIntegerField: (OFTInteger, OFTReal, OFTString), models.PositiveIntegerField: (OFTInteger, OFTReal, OFTString), models.PositiveSmallIntegerField: (OFTInteger, OFTReal, OFTString)}

    def __init__(self, model, data, mapping, layer=0, source_srs=None, encoding='utf-8', transaction_mode='commit_on_success', transform=True, unique=None, using=None):
        if False:
            i = 10
            return i + 15
        '\n        A LayerMapping object is initialized using the given Model (not an instance),\n        a DataSource (or string path to an OGR-supported data file), and a mapping\n        dictionary.  See the module level docstring for more details and keyword\n        argument usage.\n        '
        if isinstance(data, (str, Path)):
            self.ds = DataSource(data, encoding=encoding)
        else:
            self.ds = data
        self.layer = self.ds[layer]
        self.using = using if using is not None else router.db_for_write(model)
        connection = connections[self.using]
        self.spatial_backend = connection.ops
        self.mapping = mapping
        self.model = model
        self.check_layer()
        if connection.features.supports_transform:
            self.geo_field = self.geometry_field()
        else:
            transform = False
        if transform:
            self.source_srs = self.check_srs(source_srs)
            self.transform = self.coord_transform()
        else:
            self.transform = transform
        if encoding:
            from codecs import lookup
            lookup(encoding)
            self.encoding = encoding
        else:
            self.encoding = None
        if unique:
            self.check_unique(unique)
            transaction_mode = 'autocommit'
            self.unique = unique
        else:
            self.unique = None
        self.transaction_mode = transaction_mode
        if transaction_mode == 'autocommit':
            self.transaction_decorator = None
        elif transaction_mode == 'commit_on_success':
            self.transaction_decorator = transaction.atomic
        else:
            raise LayerMapError('Unrecognized transaction mode: %s' % transaction_mode)

    def check_fid_range(self, fid_range):
        if False:
            return 10
        'Check the `fid_range` keyword.'
        if fid_range:
            if isinstance(fid_range, (tuple, list)):
                return slice(*fid_range)
            elif isinstance(fid_range, slice):
                return fid_range
            else:
                raise TypeError
        else:
            return None

    def check_layer(self):
        if False:
            while True:
                i = 10
        "\n        Check the Layer metadata and ensure that it's compatible with the\n        mapping information and model. Unlike previous revisions, there is no\n        need to increment through each feature in the Layer.\n        "
        self.geom_field = False
        self.fields = {}
        ogr_fields = self.layer.fields
        ogr_field_types = self.layer.field_types

        def check_ogr_fld(ogr_map_fld):
            if False:
                i = 10
                return i + 15
            try:
                idx = ogr_fields.index(ogr_map_fld)
            except ValueError:
                raise LayerMapError('Given mapping OGR field "%s" not found in OGR Layer.' % ogr_map_fld)
            return idx
        for (field_name, ogr_name) in self.mapping.items():
            try:
                model_field = self.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                raise LayerMapError('Given mapping field "%s" not in given Model fields.' % field_name)
            fld_name = model_field.__class__.__name__
            if isinstance(model_field, GeometryField):
                if self.geom_field:
                    raise LayerMapError('LayerMapping does not support more than one GeometryField per model.')
                coord_dim = model_field.dim
                try:
                    if coord_dim == 3:
                        gtype = OGRGeomType(ogr_name + '25D')
                    else:
                        gtype = OGRGeomType(ogr_name)
                except GDALException:
                    raise LayerMapError('Invalid mapping for GeometryField "%s".' % field_name)
                ltype = self.layer.geom_type
                if not (ltype.name.startswith(gtype.name) or self.make_multi(ltype, model_field)):
                    raise LayerMapError('Invalid mapping geometry; model has %s%s, layer geometry type is %s.' % (fld_name, '(dim=3)' if coord_dim == 3 else '', ltype))
                self.geom_field = field_name
                self.coord_dim = coord_dim
                fields_val = model_field
            elif isinstance(model_field, models.ForeignKey):
                if isinstance(ogr_name, dict):
                    rel_model = model_field.remote_field.model
                    for (rel_name, ogr_field) in ogr_name.items():
                        idx = check_ogr_fld(ogr_field)
                        try:
                            rel_model._meta.get_field(rel_name)
                        except FieldDoesNotExist:
                            raise LayerMapError('ForeignKey mapping field "%s" not in %s fields.' % (rel_name, rel_model.__class__.__name__))
                    fields_val = rel_model
                else:
                    raise TypeError('ForeignKey mapping must be of dictionary type.')
            else:
                if model_field.__class__ not in self.FIELD_TYPES:
                    raise LayerMapError('Django field type "%s" has no OGR mapping (yet).' % fld_name)
                idx = check_ogr_fld(ogr_name)
                ogr_field = ogr_field_types[idx]
                if not issubclass(ogr_field, self.FIELD_TYPES[model_field.__class__]):
                    raise LayerMapError('OGR field "%s" (of type %s) cannot be mapped to Django %s.' % (ogr_field, ogr_field.__name__, fld_name))
                fields_val = model_field
            self.fields[field_name] = fields_val

    def check_srs(self, source_srs):
        if False:
            return 10
        'Check the compatibility of the given spatial reference object.'
        if isinstance(source_srs, SpatialReference):
            sr = source_srs
        elif isinstance(source_srs, self.spatial_backend.spatial_ref_sys()):
            sr = source_srs.srs
        elif isinstance(source_srs, (int, str)):
            sr = SpatialReference(source_srs)
        else:
            sr = self.layer.srs
        if not sr:
            raise LayerMapError('No source reference system defined.')
        else:
            return sr

    def check_unique(self, unique):
        if False:
            return 10
        'Check the `unique` keyword parameter -- may be a sequence or string.'
        if isinstance(unique, (list, tuple)):
            for attr in unique:
                if attr not in self.mapping:
                    raise ValueError
        elif isinstance(unique, str):
            if unique not in self.mapping:
                raise ValueError
        else:
            raise TypeError('Unique keyword argument must be set with a tuple, list, or string.')

    def feature_kwargs(self, feat):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given an OGR Feature, return a dictionary of keyword arguments for\n        constructing the mapped model.\n        '
        kwargs = {}
        for (field_name, ogr_name) in self.mapping.items():
            model_field = self.fields[field_name]
            if isinstance(model_field, GeometryField):
                try:
                    val = self.verify_geom(feat.geom, model_field)
                except GDALException:
                    raise LayerMapError('Could not retrieve geometry from feature.')
            elif isinstance(model_field, models.base.ModelBase):
                val = self.verify_fk(feat, model_field, ogr_name)
            else:
                val = self.verify_ogr_field(feat[ogr_name], model_field)
            kwargs[field_name] = val
        return kwargs

    def unique_kwargs(self, kwargs):
        if False:
            return 10
        '\n        Given the feature keyword arguments (from `feature_kwargs`), construct\n        and return the uniqueness keyword arguments -- a subset of the feature\n        kwargs.\n        '
        if isinstance(self.unique, str):
            return {self.unique: kwargs[self.unique]}
        else:
            return {fld: kwargs[fld] for fld in self.unique}

    def verify_ogr_field(self, ogr_field, model_field):
        if False:
            return 10
        '\n        Verify if the OGR Field contents are acceptable to the model field. If\n        they are, return the verified value, otherwise raise an exception.\n        '
        if isinstance(ogr_field, OFTString) and isinstance(model_field, (models.CharField, models.TextField)):
            if self.encoding and ogr_field.value is not None:
                val = force_str(ogr_field.value, self.encoding)
            else:
                val = ogr_field.value
            if model_field.max_length and val is not None and (len(val) > model_field.max_length):
                raise InvalidString('%s model field maximum string length is %s, given %s characters.' % (model_field.name, model_field.max_length, len(val)))
        elif isinstance(ogr_field, OFTReal) and isinstance(model_field, models.DecimalField):
            try:
                d = Decimal(str(ogr_field.value))
            except DecimalInvalidOperation:
                raise InvalidDecimal('Could not construct decimal from: %s' % ogr_field.value)
            dtup = d.as_tuple()
            digits = dtup[1]
            d_idx = dtup[2]
            max_prec = model_field.max_digits - model_field.decimal_places
            if d_idx < 0:
                n_prec = len(digits[:d_idx])
            else:
                n_prec = len(digits) + d_idx
            if n_prec > max_prec:
                raise InvalidDecimal('A DecimalField with max_digits %d, decimal_places %d must round to an absolute value less than 10^%d.' % (model_field.max_digits, model_field.decimal_places, max_prec))
            val = d
        elif isinstance(ogr_field, (OFTReal, OFTString)) and isinstance(model_field, models.IntegerField):
            try:
                val = int(ogr_field.value)
            except ValueError:
                raise InvalidInteger('Could not construct integer from: %s' % ogr_field.value)
        else:
            val = ogr_field.value
        return val

    def verify_fk(self, feat, rel_model, rel_mapping):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given an OGR Feature, the related model and its dictionary mapping,\n        retrieve the related model for the ForeignKey mapping.\n        '
        fk_kwargs = {}
        for (field_name, ogr_name) in rel_mapping.items():
            fk_kwargs[field_name] = self.verify_ogr_field(feat[ogr_name], rel_model._meta.get_field(field_name))
        try:
            return rel_model.objects.using(self.using).get(**fk_kwargs)
        except ObjectDoesNotExist:
            raise MissingForeignKey('No ForeignKey %s model found with keyword arguments: %s' % (rel_model.__name__, fk_kwargs))

    def verify_geom(self, geom, model_field):
        if False:
            while True:
                i = 10
        '\n        Verify the geometry -- construct and return a GeometryCollection\n        if necessary (for example if the model field is MultiPolygonField while\n        the mapped shapefile only contains Polygons).\n        '
        if self.coord_dim != geom.coord_dim:
            geom.coord_dim = self.coord_dim
        if self.make_multi(geom.geom_type, model_field):
            multi_type = self.MULTI_TYPES[geom.geom_type.num]
            g = OGRGeometry(multi_type)
            g.add(geom)
        else:
            g = geom
        if self.transform:
            g.transform(self.transform)
        return g.wkt

    def coord_transform(self):
        if False:
            i = 10
            return i + 15
        'Return the coordinate transformation object.'
        SpatialRefSys = self.spatial_backend.spatial_ref_sys()
        try:
            target_srs = SpatialRefSys.objects.using(self.using).get(srid=self.geo_field.srid).srs
            return CoordTransform(self.source_srs, target_srs)
        except Exception as exc:
            raise LayerMapError('Could not translate between the data source and model geometry.') from exc

    def geometry_field(self):
        if False:
            i = 10
            return i + 15
        'Return the GeometryField instance associated with the geographic column.'
        opts = self.model._meta
        return opts.get_field(self.geom_field)

    def make_multi(self, geom_type, model_field):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given the OGRGeomType for a geometry and its associated GeometryField,\n        determine whether the geometry should be turned into a GeometryCollection.\n        '
        return geom_type.num in self.MULTI_TYPES and model_field.__class__.__name__ == 'Multi%s' % geom_type.django

    def save(self, verbose=False, fid_range=False, step=False, progress=False, silent=False, stream=sys.stdout, strict=False):
        if False:
            i = 10
            return i + 15
        "\n        Save the contents from the OGR DataSource Layer into the database\n        according to the mapping dictionary given at initialization.\n\n        Keyword Parameters:\n         verbose:\n           If set, information will be printed subsequent to each model save\n           executed on the database.\n\n         fid_range:\n           May be set with a slice or tuple of (begin, end) feature ID's to map\n           from the data source.  In other words, this keyword enables the user\n           to selectively import a subset range of features in the geographic\n           data source.\n\n         step:\n           If set with an integer, transactions will occur at every step\n           interval. For example, if step=1000, a commit would occur after\n           the 1,000th feature, the 2,000th feature etc.\n\n         progress:\n           When this keyword is set, status information will be printed giving\n           the number of features processed and successfully saved.  By default,\n           progress information will pe printed every 1000 features processed,\n           however, this default may be overridden by setting this keyword with an\n           integer for the desired interval.\n\n         stream:\n           Status information will be written to this file handle.  Defaults to\n           using `sys.stdout`, but any object with a `write` method is supported.\n\n         silent:\n           By default, non-fatal error notifications are printed to stdout, but\n           this keyword may be set to disable these notifications.\n\n         strict:\n           Execution of the model mapping will cease upon the first error\n           encountered.  The default behavior is to attempt to continue.\n        "
        default_range = self.check_fid_range(fid_range)
        if progress:
            if progress is True or not isinstance(progress, int):
                progress_interval = 1000
            else:
                progress_interval = progress

        def _save(feat_range=default_range, num_feat=0, num_saved=0):
            if False:
                print('Hello World!')
            if feat_range:
                layer_iter = self.layer[feat_range]
            else:
                layer_iter = self.layer
            for feat in layer_iter:
                num_feat += 1
                try:
                    kwargs = self.feature_kwargs(feat)
                except LayerMapError as msg:
                    if strict:
                        raise
                    elif not silent:
                        stream.write('Ignoring Feature ID %s because: %s\n' % (feat.fid, msg))
                else:
                    is_update = False
                    if self.unique:
                        try:
                            u_kwargs = self.unique_kwargs(kwargs)
                            m = self.model.objects.using(self.using).get(**u_kwargs)
                            is_update = True
                            geom_value = getattr(m, self.geom_field)
                            if geom_value is None:
                                geom = OGRGeometry(kwargs[self.geom_field])
                            else:
                                geom = geom_value.ogr
                                new = OGRGeometry(kwargs[self.geom_field])
                                for g in new:
                                    geom.add(g)
                            setattr(m, self.geom_field, geom.wkt)
                        except ObjectDoesNotExist:
                            m = self.model(**kwargs)
                    else:
                        m = self.model(**kwargs)
                    try:
                        m.save(using=self.using)
                        num_saved += 1
                        if verbose:
                            stream.write('%s: %s\n' % ('Updated' if is_update else 'Saved', m))
                    except Exception as msg:
                        if strict:
                            if not silent:
                                stream.write('Failed to save the feature (id: %s) into the model with the keyword arguments:\n' % feat.fid)
                                stream.write('%s\n' % kwargs)
                            raise
                        elif not silent:
                            stream.write('Failed to save %s:\n %s\nContinuing\n' % (kwargs, msg))
                if progress and num_feat % progress_interval == 0:
                    stream.write('Processed %d features, saved %d ...\n' % (num_feat, num_saved))
            return (num_saved, num_feat)
        if self.transaction_decorator is not None:
            _save = self.transaction_decorator(_save)
        nfeat = self.layer.num_feat
        if step and isinstance(step, int) and (step < nfeat):
            if default_range:
                raise LayerMapError('The `step` keyword may not be used in conjunction with the `fid_range` keyword.')
            (beg, num_feat, num_saved) = (0, 0, 0)
            indices = range(step, nfeat, step)
            n_i = len(indices)
            for (i, end) in enumerate(indices):
                if i + 1 == n_i:
                    step_slice = slice(beg, None)
                else:
                    step_slice = slice(beg, end)
                try:
                    (num_feat, num_saved) = _save(step_slice, num_feat, num_saved)
                    beg = end
                except Exception:
                    stream.write('%s\nFailed to save slice: %s\n' % ('=-' * 20, step_slice))
                    raise
        else:
            _save()