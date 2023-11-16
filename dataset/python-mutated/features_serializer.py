import json
from featuretools.primitives.utils import serialize_primitive
from featuretools.utils.s3_utils import get_transport_params, use_smartopen_features
from featuretools.utils.wrangle import _is_s3, _is_url
from featuretools.version import FEATURES_SCHEMA_VERSION
from featuretools.version import __version__ as ft_version

def save_features(features, location=None, profile_name=None):
    if False:
        for i in range(10):
            print('nop')
    'Saves the features list as JSON to a specified filepath/S3 path, writes to an open file, or\n    returns the serialized features as a JSON string. If no file provided, returns a string.\n\n    Args:\n        features (list[:class:`.FeatureBase`]): List of Feature definitions.\n\n        location (str or :class:`.FileObject`, optional): The location of where to save\n            the features list which must include the name of the file,\n            or a writeable file handle to write to. If location is None, will return a JSON string\n            of the serialized features.\n            Default: None\n\n        profile_name (str, bool): The AWS profile specified to write to S3. Will default to None and search for AWS credentials.\n                                    Set to False to use an anonymous profile.\n\n    Note:\n        Features saved in one version of Featuretools are not guaranteed to work in another.\n        After upgrading Featuretools, features may need to be generated again.\n\n    Example:\n        .. ipython:: python\n            :suppress:\n\n            from featuretools.tests.testing_utils import (\n                make_ecommerce_entityset)\n            import featuretools as ft\n            es = make_ecommerce_entityset()\n            import os\n\n        .. code-block:: python\n\n            f1 = ft.Feature(es["log"].ww["product_id"])\n            f2 = ft.Feature(es["log"].ww["purchased"])\n            f3 = ft.Feature(es["log"].ww["value"])\n\n            features = [f1, f2, f3]\n\n            # Option 1\n            filepath = os.path.join(\'/Home/features/\', \'list.json\')\n            ft.save_features(features, filepath)\n\n            # Option 2\n            filepath = os.path.join(\'/Home/features/\', \'list.json\')\n            with open(filepath, \'w\') as f:\n                ft.save_features(features, f)\n\n            # Option 3\n            features_string = ft.save_features(features)\n    .. seealso::\n        :func:`.load_features`\n    '
    return FeaturesSerializer(features).save(location, profile_name=profile_name)

class FeaturesSerializer(object):

    def __init__(self, feature_list):
        if False:
            while True:
                i = 10
        self.feature_list = feature_list
        self._features_dict = None

    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        names_list = [feat.unique_name() for feat in self.feature_list]
        es = self.feature_list[0].entityset
        (feature_defs, primitive_defs) = self._feature_definitions()
        return {'schema_version': FEATURES_SCHEMA_VERSION, 'ft_version': ft_version, 'entityset': es.to_dictionary(), 'feature_list': names_list, 'feature_definitions': feature_defs, 'primitive_definitions': primitive_defs}

    def save(self, location, profile_name):
        if False:
            return 10
        features_dict = self.to_dict()
        if location is None:
            return json.dumps(features_dict)
        if isinstance(location, str):
            if _is_url(location):
                raise ValueError('Writing to URLs is not supported')
            if _is_s3(location):
                transport_params = get_transport_params(profile_name)
                use_smartopen_features(location, features_dict, transport_params, read=False)
            else:
                with open(location, 'w') as f:
                    json.dump(features_dict, f)
        else:
            json.dump(features_dict, location)

    def _feature_definitions(self):
        if False:
            print('Hello World!')
        if not self._features_dict:
            self._features_dict = {}
            self._primitives_dict = {}
            for feature in self.feature_list:
                self._serialize_feature(feature)
            primitive_number = 0
            primitive_id_to_key = {}
            for (name, feature) in self._features_dict.items():
                primitive = feature['arguments'].get('primitive')
                if primitive:
                    primitive_id = id(primitive)
                    if primitive_id not in primitive_id_to_key.keys():
                        primitives_dict_key = str(primitive_number)
                        primitive_id_to_key[primitive_id] = primitives_dict_key
                        self._primitives_dict[primitives_dict_key] = serialize_primitive(primitive)
                        self._features_dict[name]['arguments']['primitive'] = primitives_dict_key
                        primitive_number += 1
                    else:
                        key = primitive_id_to_key[primitive_id]
                        self._features_dict[name]['arguments']['primitive'] = key
        return (self._features_dict, self._primitives_dict)

    def _serialize_feature(self, feature):
        if False:
            i = 10
            return i + 15
        name = feature.unique_name()
        if name not in self._features_dict:
            self._features_dict[feature.unique_name()] = feature.to_dictionary()
            for dependency in feature.get_dependencies(deep=True):
                name = dependency.unique_name()
                if name not in self._features_dict:
                    self._features_dict[name] = dependency.to_dictionary()