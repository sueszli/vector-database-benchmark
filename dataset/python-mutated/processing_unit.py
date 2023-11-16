from __future__ import unicode_literals
import io
import json
import shutil
from abc import ABCMeta, abstractmethod, abstractproperty
from builtins import str, bytes
from pathlib import Path
from future.utils import with_metaclass
from snips_nlu.common.abc_utils import abstractclassmethod, classproperty
from snips_nlu.common.io_utils import temp_dir, unzip_archive
from snips_nlu.common.registrable import Registrable
from snips_nlu.common.utils import json_string, check_random_state
from snips_nlu.constants import BUILTIN_ENTITY_PARSER, CUSTOM_ENTITY_PARSER, CUSTOM_ENTITY_PARSER_USAGE, RESOURCES, LANGUAGE, RANDOM_STATE
from snips_nlu.entity_parser import BuiltinEntityParser, CustomEntityParser, CustomEntityParserUsage
from snips_nlu.exceptions import LoadingError
from snips_nlu.pipeline.configs import ProcessingUnitConfig
from snips_nlu.pipeline.configs.config import DefaultProcessingUnitConfig
from snips_nlu.resources import load_resources

class ProcessingUnit(with_metaclass(ABCMeta, Registrable)):
    """Abstraction of a NLU pipeline unit

    Pipeline processing units such as intent parsers, intent classifiers and
    slot fillers must implement this class.

     A :class:`ProcessingUnit` is associated with a *config_type*, which
    represents the :class:`.ProcessingUnitConfig` used to initialize it.
    """

    def __init__(self, config, **shared):
        if False:
            return 10
        if config is None:
            self.config = self.default_config()
        elif isinstance(config, ProcessingUnitConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = self.config_type.from_dict(config)
        else:
            raise ValueError('Unexpected config type: %s' % type(config))
        if self.config is not None:
            self.config.set_unit_name(self.unit_name)
        self.builtin_entity_parser = shared.get(BUILTIN_ENTITY_PARSER)
        self.custom_entity_parser = shared.get(CUSTOM_ENTITY_PARSER)
        self.resources = shared.get(RESOURCES)
        self.random_state = check_random_state(shared.get(RANDOM_STATE))

    @classproperty
    def config_type(cls):
        if False:
            while True:
                i = 10
        return DefaultProcessingUnitConfig

    @classmethod
    def default_config(cls):
        if False:
            print('Hello World!')
        config = cls.config_type()
        config.set_unit_name(cls.unit_name)
        return config

    @classproperty
    def unit_name(cls):
        if False:
            for i in range(10):
                print('nop')
        return ProcessingUnit.registered_name(cls)

    @classmethod
    def from_config(cls, unit_config, **shared):
        if False:
            i = 10
            return i + 15
        'Build a :class:`ProcessingUnit` from the provided config'
        unit = cls.by_name(unit_config.unit_name)
        return unit(unit_config, **shared)

    @classmethod
    def load_from_path(cls, unit_path, unit_name=None, **shared):
        if False:
            while True:
                i = 10
        'Load a :class:`ProcessingUnit` from a persisted processing unit\n        directory\n\n        Args:\n            unit_path (str or :class:`pathlib.Path`): path to the persisted\n                processing unit\n            unit_name (str, optional): Name of the processing unit to load.\n                By default, the unit name is assumed to be stored in a\n                "metadata.json" file located in the directory at unit_path.\n\n        Raises:\n            LoadingError: when unit_name is None and no metadata file is found\n                in the processing unit directory\n        '
        unit_path = Path(unit_path)
        if unit_name is None:
            metadata_path = unit_path / 'metadata.json'
            if not metadata_path.exists():
                raise LoadingError('Missing metadata for processing unit at path %s' % str(unit_path))
            with metadata_path.open(encoding='utf8') as f:
                metadata = json.load(f)
            unit_name = metadata['unit_name']
        unit = cls.by_name(unit_name)
        return unit.from_path(unit_path, **shared)

    @classmethod
    def get_config(cls, unit_config):
        if False:
            while True:
                i = 10
        'Returns the :class:`.ProcessingUnitConfig` corresponding to\n        *unit_config*'
        if isinstance(unit_config, ProcessingUnitConfig):
            return unit_config
        elif isinstance(unit_config, dict):
            unit_name = unit_config['unit_name']
            processing_unit_type = cls.by_name(unit_name)
            return processing_unit_type.config_type.from_dict(unit_config)
        elif isinstance(unit_config, (str, bytes)):
            unit_name = unit_config
            unit_config = {'unit_name': unit_name}
            processing_unit_type = cls.by_name(unit_name)
            return processing_unit_type.config_type.from_dict(unit_config)
        else:
            raise ValueError('Expected `unit_config` to be an instance of ProcessingUnitConfig or dict or str but found: %s' % type(unit_config))

    @abstractproperty
    def fitted(self):
        if False:
            return 10
        'Whether or not the processing unit has already been trained'
        pass

    def load_resources_if_needed(self, language):
        if False:
            i = 10
            return i + 15
        if self.resources is None or self.fitted:
            required_resources = None
            if self.config is not None:
                required_resources = self.config.get_required_resources()
            self.resources = load_resources(language, required_resources)

    def fit_builtin_entity_parser_if_needed(self, dataset):
        if False:
            return 10
        if self.builtin_entity_parser is None or self.fitted:
            self.builtin_entity_parser = BuiltinEntityParser.build(dataset=dataset)
        return self

    def fit_custom_entity_parser_if_needed(self, dataset):
        if False:
            print('Hello World!')
        required_resources = self.config.get_required_resources()
        if not required_resources or not required_resources.get(CUSTOM_ENTITY_PARSER_USAGE):
            parser_usage = CustomEntityParserUsage.WITHOUT_STEMS
        else:
            parser_usage = required_resources[CUSTOM_ENTITY_PARSER_USAGE]
        if self.custom_entity_parser is None or self.fitted:
            self.load_resources_if_needed(dataset[LANGUAGE])
            self.custom_entity_parser = CustomEntityParser.build(dataset, parser_usage, self.resources)
        return self

    def persist_metadata(self, path, **kwargs):
        if False:
            print('Hello World!')
        metadata = {'unit_name': self.unit_name}
        metadata.update(kwargs)
        metadata_json = json_string(metadata)
        with (path / 'metadata.json').open(mode='w', encoding='utf8') as f:
            f.write(metadata_json)

    @abstractmethod
    def persist(self, path):
        if False:
            print('Hello World!')
        pass

    @abstractclassmethod
    def from_path(cls, path, **shared):
        if False:
            print('Hello World!')
        pass

    def to_byte_array(self):
        if False:
            for i in range(10):
                print('nop')
        'Serialize the :class:`ProcessingUnit` instance into a bytearray\n\n        This method persists the processing unit in a temporary directory, zip\n        the directory and return the zipped file as binary data.\n\n        Returns:\n            bytearray: the processing unit as bytearray data\n        '
        cleaned_unit_name = _sanitize_unit_name(self.unit_name)
        with temp_dir() as tmp_dir:
            processing_unit_dir = tmp_dir / cleaned_unit_name
            self.persist(processing_unit_dir)
            archive_base_name = tmp_dir / cleaned_unit_name
            archive_name = archive_base_name.with_suffix('.zip')
            shutil.make_archive(base_name=str(archive_base_name), format='zip', root_dir=str(tmp_dir), base_dir=cleaned_unit_name)
            with archive_name.open(mode='rb') as f:
                processing_unit_bytes = bytearray(f.read())
        return processing_unit_bytes

    @classmethod
    def from_byte_array(cls, unit_bytes, **shared):
        if False:
            i = 10
            return i + 15
        'Load a :class:`ProcessingUnit` instance from a bytearray\n\n        Args:\n            unit_bytes (bytearray): A bytearray representing a zipped\n                processing unit.\n        '
        cleaned_unit_name = _sanitize_unit_name(cls.unit_name)
        with temp_dir() as tmp_dir:
            file_io = io.BytesIO(unit_bytes)
            unzip_archive(file_io, str(tmp_dir))
            processing_unit = cls.from_path(tmp_dir / cleaned_unit_name, **shared)
        return processing_unit

def _sanitize_unit_name(unit_name):
    if False:
        print('Hello World!')
    return unit_name.lower().strip().replace(' ', '').replace('/', '').replace('\\', '')