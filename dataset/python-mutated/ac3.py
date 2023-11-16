import mutagen
from picard import log
from picard.config import get_config
from picard.formats.apev2 import APEv2File
from picard.util import encode_filename
from .mutagenext import ac3

class AC3File(APEv2File):
    EXTENSIONS = ['.ac3', '.eac3']
    NAME = 'AC-3'
    _File = ac3.AC3APEv2

    def _info(self, metadata, file):
        if False:
            while True:
                i = 10
        super()._info(metadata, file)
        if hasattr(file.info, 'codec') and file.info.codec == 'ec-3':
            format = 'Enhanced AC-3'
        else:
            format = self.NAME
        if file.tags:
            metadata['~format'] = '%s (APEv2)' % format
        else:
            metadata['~format'] = format

    def _save(self, filename, metadata):
        if False:
            while True:
                i = 10
        config = get_config()
        if config.setting['ac3_save_ape']:
            super()._save(filename, metadata)
        elif config.setting['remove_ape_from_ac3']:
            try:
                mutagen.apev2.delete(encode_filename(filename))
            except BaseException:
                log.exception('Error removing APEv2 tags from %s', filename)

    @classmethod
    def supports_tag(cls, name):
        if False:
            return 10
        config = get_config()
        if config.setting['ac3_save_ape']:
            return APEv2File.supports_tag(name)
        else:
            return False