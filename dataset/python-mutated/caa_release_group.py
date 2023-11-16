from picard.coverart.image import CaaCoverArtImage, CaaThumbnailCoverArtImage
from picard.coverart.providers.caa import CoverArtProviderCaa

class CaaCoverArtImageRg(CaaCoverArtImage):
    pass

class CaaThumbnailCoverArtImageRg(CaaThumbnailCoverArtImage):
    pass

class CoverArtProviderCaaReleaseGroup(CoverArtProviderCaa):
    """Use cover art from album release group"""
    NAME = 'CaaReleaseGroup'
    TITLE = N_('Cover Art Archive: Release Group')
    OPTIONS = None
    ignore_json_not_found_error = True
    coverartimage_class = CaaCoverArtImageRg
    coverartimage_thumbnail_class = CaaThumbnailCoverArtImageRg

    def enabled(self):
        if False:
            i = 10
            return i + 15
        return not self.coverart.front_image_found

    @property
    def _caa_path(self):
        if False:
            i = 10
            return i + 15
        return '/release-group/%s/' % self.metadata['musicbrainz_releasegroupid']