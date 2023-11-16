from picard import log
from picard.coverart.image import CoverArtImage
from picard.coverart.providers.provider import CoverArtProvider

class CoverArtProviderUrlRelationships(CoverArtProvider):
    """Use cover art link and has_cover_art_at MusicBrainz relationships to get
    cover art"""
    NAME = 'UrlRelationships'
    TITLE = N_('Allowed Cover Art URLs')

    def queue_images(self):
        if False:
            for i in range(10):
                print('nop')
        self.match_url_relations(('cover art link', 'has_cover_art_at'), self._queue_from_relationship)
        return CoverArtProvider.FINISHED

    def _queue_from_relationship(self, url):
        if False:
            print('Hello World!')
        log.debug('Found cover art link in URL relationship')
        self.queue_put(CoverArtImage(url))