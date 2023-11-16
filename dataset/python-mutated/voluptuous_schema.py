import difflib
import itertools
import voluptuous as vol
from esphome.schema_extractors import schema_extractor_extended

class ExtraKeysInvalid(vol.Invalid):

    def __init__(self, *arg, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.candidates = kwargs.pop('candidates')
        vol.Invalid.__init__(self, *arg, **kwargs)

def ensure_multiple_invalid(err):
    if False:
        while True:
            i = 10
    if isinstance(err, vol.MultipleInvalid):
        return err
    return vol.MultipleInvalid(err)

class _Schema(vol.Schema):
    """Custom cv.Schema that prints similar keys on error."""

    def __init__(self, schema, required=False, extra=vol.PREVENT_EXTRA, extra_schemas=None):
        if False:
            print('Hello World!')
        super().__init__(schema, required=required, extra=extra)
        self._extra_schemas = extra_schemas or []

    def __call__(self, data):
        if False:
            i = 10
            return i + 15
        res = super().__call__(data)
        for extra in self._extra_schemas:
            try:
                res = extra(res)
            except vol.Invalid as err:
                raise ensure_multiple_invalid(err)
        return res

    def _compile_mapping(self, schema, invalid_msg=None):
        if False:
            print('Hello World!')
        invalid_msg = invalid_msg or 'mapping value'
        for key in schema:
            if key is vol.Extra:
                raise ValueError('ESPHome does not allow vol.Extra')
            if isinstance(key, vol.Remove):
                raise ValueError('ESPHome does not allow vol.Remove')
            if isinstance(key, vol.primitive_types):
                raise ValueError('All schema keys must be wrapped in cv.Required or cv.Optional')
        all_required_keys = {key for key in schema if isinstance(key, vol.Required)}
        all_default_keys = [key for key in schema if isinstance(key, vol.Optional)]
        _compiled_schema = {}
        for (skey, svalue) in vol.iteritems(schema):
            new_key = self._compile(skey)
            new_value = self._compile(svalue)
            _compiled_schema[skey] = (new_key, new_value)
        candidates = list(vol.schema_builder._iterate_mapping_candidates(_compiled_schema))
        additional_candidates = []
        candidates_by_key = {}
        for (skey, (ckey, cvalue)) in candidates:
            if type(skey) in vol.primitive_types:
                candidates_by_key.setdefault(skey, []).append((skey, (ckey, cvalue)))
            elif isinstance(skey, vol.Marker) and type(skey.schema) in vol.primitive_types:
                candidates_by_key.setdefault(skey.schema, []).append((skey, (ckey, cvalue)))
            else:
                additional_candidates.append((skey, (ckey, cvalue)))
        key_names = []
        for skey in schema:
            if isinstance(skey, str):
                key_names.append(skey)
            elif isinstance(skey, vol.Marker) and isinstance(skey.schema, str):
                key_names.append(skey.schema)

        def validate_mapping(path, iterable, out):
            if False:
                i = 10
                return i + 15
            required_keys = all_required_keys.copy()
            key_value_map = type(out)()
            for (key, value) in iterable:
                key_value_map[key] = value
            for key in all_default_keys:
                if not isinstance(key.default, vol.Undefined) and key.schema not in key_value_map:
                    key_value_map[key.schema] = key.default()
            error = None
            errors = []
            for (key, value) in key_value_map.items():
                key_path = path + [key]
                relevant_candidates = itertools.chain(candidates_by_key.get(key, []), additional_candidates)
                for (skey, (ckey, cvalue)) in relevant_candidates:
                    try:
                        new_key = ckey(key_path, key)
                    except vol.Invalid as e:
                        if len(e.path) > len(key_path):
                            raise
                        if not error or len(e.path) > len(error.path):
                            error = e
                        continue
                    exception_errors = []
                    try:
                        cval = cvalue(key_path, value)
                        out[new_key] = cval
                    except vol.MultipleInvalid as e:
                        exception_errors.extend(e.errors)
                    except vol.Invalid as e:
                        exception_errors.append(e)
                    if exception_errors:
                        for err in exception_errors:
                            if len(err.path) <= len(key_path):
                                err.error_type = invalid_msg
                            errors.append(err)
                        required_keys.discard(skey)
                        break
                    required_keys.discard(skey)
                    break
                else:
                    if self.extra == vol.ALLOW_EXTRA:
                        out[key] = value
                    elif self.extra != vol.REMOVE_EXTRA:
                        if isinstance(key, str) and key_names:
                            matches = difflib.get_close_matches(key, key_names)
                            errors.append(ExtraKeysInvalid('extra keys not allowed', key_path, candidates=matches))
                        else:
                            errors.append(vol.Invalid('extra keys not allowed', key_path))
            for key in required_keys:
                msg = getattr(key, 'msg', None) or 'required key not provided'
                errors.append(vol.RequiredFieldInvalid(msg, path + [key]))
            if errors:
                raise vol.MultipleInvalid(errors)
            return out
        return validate_mapping

    def add_extra(self, validator):
        if False:
            return 10
        validator = _Schema(validator)
        self._extra_schemas.append(validator)
        return self

    def prepend_extra(self, validator):
        if False:
            return 10
        validator = _Schema(validator)
        self._extra_schemas.insert(0, validator)
        return self

    @schema_extractor_extended
    def extend(self, *schemas, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        extra = kwargs.pop('extra', None)
        if kwargs:
            raise ValueError
        if not schemas:
            return self.extend({})
        if len(schemas) != 1:
            ret = self
            for schema in schemas:
                ret = ret.extend(schema)
            return ret
        schema = schemas[0]
        if isinstance(schema, vol.Schema):
            schema = schema.schema
        ret = super().extend(schema, extra=extra)
        return _Schema(ret.schema, extra=ret.extra, extra_schemas=self._extra_schemas)