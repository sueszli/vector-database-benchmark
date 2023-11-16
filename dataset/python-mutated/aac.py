from mutagen._util import loadfile
from mutagen.aac import AAC
from mutagen.apev2 import APENoHeaderError, APEv2, _APEv2Data, error as APEError

class AACAPEv2(AAC):
    """AAC file with APEv2 tags.
    """

    @loadfile()
    def load(self, filething):
        if False:
            return 10
        super().load(filething)
        try:
            self.tags = APEv2(filething)
            if not hasattr(self.info, 'bitrate') or self.info.bitrate == 0:
                return
            ape_data = _APEv2Data(filething.fileobj)
            if ape_data.size is not None:
                extra_length = 8.0 * ape_data.size / self.info.bitrate
                self.info.length = max(self.info.length - extra_length, 0.001)
        except APENoHeaderError:
            self.tags = None

    def add_tags(self):
        if False:
            i = 10
            return i + 15
        if self.tags is None:
            self.tags = APEv2()
        else:
            raise APEError('%r already has tags: %r' % (self, self.tags))