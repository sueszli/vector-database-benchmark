from collections import OrderedDict, namedtuple
from PyQt6.QtNetwork import QNetworkReply, QNetworkRequest
from picard import log
from picard.config import BoolOption, IntOption, ListOption, get_config
from picard.const import CAA_URL
from picard.coverart.image import CaaCoverArtImage, CaaThumbnailCoverArtImage
from picard.coverart.providers.provider import CoverArtProvider, ProviderOptions
from picard.coverart.utils import CAA_TYPES, translate_caa_type
from picard.webservice import ratecontrol
from picard.ui.caa_types_selector import display_caa_types_selector
from picard.ui.ui_provider_options_caa import Ui_CaaOptions
CaaSizeItem = namedtuple('CaaSizeItem', ['thumbnail', 'label'])
_CAA_THUMBNAIL_SIZE_MAP = OrderedDict([(250, CaaSizeItem('250', N_('250 px'))), (500, CaaSizeItem('500', N_('500 px'))), (1200, CaaSizeItem('1200', N_('1200 px'))), (-1, CaaSizeItem(None, N_('Full size')))])
_CAA_THUMBNAIL_SIZE_ALIASES = {'500': 'large', '250': 'small'}
_CAA_IMAGE_SIZE_DEFAULT = 500
_CAA_IMAGE_TYPE_DEFAULT_INCLUDE = ['front']
_CAA_IMAGE_TYPE_DEFAULT_EXCLUDE = ['matrix/runout', 'raw/unedited', 'watermark']
ratecontrol.set_minimum_delay_for_url(CAA_URL, 0)
ratecontrol.set_minimum_delay_for_url('https://archive.org', 0)

def caa_url_fallback_list(desired_size, thumbnails):
    if False:
        for i in range(10):
            print('nop')
    'List of thumbnail urls equal or smaller than size, in size decreasing order\n    It is used for find the "best" thumbnail according to:\n        - user choice\n        - thumbnail availability\n    If user choice isn\'t matching an available thumbnail size, a fallback to\n    smaller thumbnails is possible\n    This function returns the list of possible urls, ordered from the biggest\n    matching the user choice to the smallest one.\n    Of course, if none are possible, the returned list may be empty.\n    '
    reversed_map = OrderedDict(reversed(list(_CAA_THUMBNAIL_SIZE_MAP.items())))
    urls = []
    for (item_id, item) in reversed_map.items():
        if item_id == -1 or item_id > desired_size:
            continue
        url = thumbnails.get(item.thumbnail, None)
        if url is None:
            size_alias = _CAA_THUMBNAIL_SIZE_ALIASES.get(item.thumbnail, None)
            if size_alias is not None:
                url = thumbnails.get(size_alias, None)
        if url is not None:
            urls.append(url)
    return urls

class ProviderOptionsCaa(ProviderOptions):
    """
        Options for Cover Art Archive cover art provider
    """
    TITLE = N_('Cover Art Archive')
    HELP_URL = '/config/options_cover_art_archive.html'
    options = [BoolOption('setting', 'caa_approved_only', False), IntOption('setting', 'caa_image_size', _CAA_IMAGE_SIZE_DEFAULT), ListOption('setting', 'caa_image_types', _CAA_IMAGE_TYPE_DEFAULT_INCLUDE), BoolOption('setting', 'caa_restrict_image_types', True), ListOption('setting', 'caa_image_types_to_omit', _CAA_IMAGE_TYPE_DEFAULT_EXCLUDE)]
    _options_ui = Ui_CaaOptions

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.ui.restrict_images_types.clicked.connect(self.update_caa_types)
        self.ui.select_caa_types.clicked.connect(self.select_caa_types)

    def restore_defaults(self):
        if False:
            print('Hello World!')
        self.caa_image_types = _CAA_IMAGE_TYPE_DEFAULT_INCLUDE
        self.caa_image_types_to_omit = _CAA_IMAGE_TYPE_DEFAULT_EXCLUDE
        super().restore_defaults()

    def load(self):
        if False:
            return 10
        self.ui.cb_image_size.clear()
        for (item_id, item) in _CAA_THUMBNAIL_SIZE_MAP.items():
            self.ui.cb_image_size.addItem(_(item.label), userData=item_id)
        config = get_config()
        size = config.setting['caa_image_size']
        index = self.ui.cb_image_size.findData(size)
        if index < 0:
            index = self.ui.cb_image_size.findData(_CAA_IMAGE_SIZE_DEFAULT)
        self.ui.cb_image_size.setCurrentIndex(index)
        self.ui.cb_approved_only.setChecked(config.setting['caa_approved_only'])
        self.ui.restrict_images_types.setChecked(config.setting['caa_restrict_image_types'])
        self.caa_image_types = config.setting['caa_image_types']
        self.caa_image_types_to_omit = config.setting['caa_image_types_to_omit']
        self.update_caa_types()

    def save(self):
        if False:
            return 10
        config = get_config()
        size = self.ui.cb_image_size.currentData()
        config.setting['caa_image_size'] = size
        config.setting['caa_approved_only'] = self.ui.cb_approved_only.isChecked()
        config.setting['caa_restrict_image_types'] = self.ui.restrict_images_types.isChecked()
        config.setting['caa_image_types'] = self.caa_image_types
        config.setting['caa_image_types_to_omit'] = self.caa_image_types_to_omit

    def update_caa_types(self):
        if False:
            print('Hello World!')
        enabled = self.ui.restrict_images_types.isChecked()
        self.ui.select_caa_types.setEnabled(enabled)

    def select_caa_types(self):
        if False:
            return 10
        known_types = {t['name']: translate_caa_type(t['name']) for t in CAA_TYPES}
        (types, types_to_omit, ok) = display_caa_types_selector(parent=self, types_include=self.caa_image_types, types_exclude=self.caa_image_types_to_omit, default_include=_CAA_IMAGE_TYPE_DEFAULT_INCLUDE, default_exclude=_CAA_IMAGE_TYPE_DEFAULT_EXCLUDE, known_types=known_types)
        if ok:
            self.caa_image_types = types
            self.caa_image_types_to_omit = types_to_omit

class CoverArtProviderCaa(CoverArtProvider):
    """Get cover art from Cover Art Archive using release mbid"""
    NAME = 'Cover Art Archive'
    TITLE = N_('Cover Art Archive: Release')
    OPTIONS = ProviderOptionsCaa
    ignore_json_not_found_error = False
    coverartimage_class = CaaCoverArtImage
    coverartimage_thumbnail_class = CaaThumbnailCoverArtImage

    def __init__(self, coverart):
        if False:
            i = 10
            return i + 15
        super().__init__(coverart)
        config = get_config()
        self.restrict_types = config.setting['caa_restrict_image_types']
        if self.restrict_types:
            self.included_types = {t.lower() for t in config.setting['caa_image_types']}
            self.excluded_types = {t.lower() for t in config.setting['caa_image_types_to_omit']}
            self.included_types_count = len(self.included_types)

    @property
    def _has_suitable_artwork(self):
        if False:
            return 10
        if 'cover-art-archive' not in self.release:
            log.debug('No Cover Art Archive information for %s', self.release['id'])
            return False
        caa_node = self.release['cover-art-archive']
        caa_has_suitable_artwork = caa_node['artwork']
        if not caa_has_suitable_artwork:
            log.debug('There are no images in the Cover Art Archive for %s', self.release['id'])
            return False
        if self.restrict_types:
            want_front = 'front' in self.included_types
            want_back = 'back' in self.included_types
            caa_has_front = caa_node['front']
            caa_has_back = caa_node['back']
            if self.included_types_count == 2 and (want_front or want_back):
                front_in_caa = caa_has_front or not want_front
                back_in_caa = caa_has_back or not want_back
                caa_has_suitable_artwork = front_in_caa or back_in_caa
            elif self.included_types_count == 1 and (want_front or want_back):
                front_in_caa = caa_has_front and want_front
                back_in_caa = caa_has_back and want_back
                caa_has_suitable_artwork = front_in_caa or back_in_caa
        if not caa_has_suitable_artwork:
            log.debug('There are no suitable images in the Cover Art Archive for %s', self.release['id'])
        else:
            log.debug('There are suitable images in the Cover Art Archive for %s', self.release['id'])
        return caa_has_suitable_artwork

    def enabled(self):
        if False:
            return 10
        'Check if CAA artwork has to be downloaded'
        if not super().enabled():
            return False
        if self.restrict_types and (not self.included_types_count):
            log.debug('User disabled all Cover Art Archive types')
            return False
        return self._has_suitable_artwork

    @property
    def _caa_path(self):
        if False:
            while True:
                i = 10
        return '/release/%s/' % self.metadata['musicbrainz_albumid']

    def queue_images(self):
        if False:
            return 10
        self.album.tagger.webservice.get_url(url=CAA_URL + self._caa_path, handler=self._caa_json_downloaded, priority=True, important=False, cacheloadcontrol=QNetworkRequest.CacheLoadControl.PreferNetwork)
        self.album._requests += 1
        return CoverArtProvider.WAIT

    def _caa_json_downloaded(self, data, http, error):
        if False:
            print('Hello World!')
        'Parse CAA JSON file and queue CAA cover art images for download'
        self.album._requests -= 1
        if error:
            if not (error == QNetworkReply.NetworkError.ContentNotFoundError and self.ignore_json_not_found_error):
                self.error('CAA JSON error: %s' % http.errorString())
        else:
            if self.restrict_types:
                log.debug('CAA types: included: %s, excluded: %s', self.included_types, self.excluded_types)
            try:
                config = get_config()
                for image in data['images']:
                    if config.setting['caa_approved_only'] and (not image['approved']):
                        continue
                    is_pdf = image['image'].endswith('.pdf')
                    if is_pdf and (not config.setting['save_images_to_files']):
                        log.debug('Skipping pdf cover art : %s', image['image'])
                        continue
                    if not image['types']:
                        image['types'] = ['unknown']
                    else:
                        image['types'] = [t.lower() for t in image['types']]
                    if self.restrict_types:
                        accepted = bool(set(image['types']).intersection(self.included_types).difference(self.excluded_types))
                        log.debug('CAA image %s: %s  %s', 'accepted' if accepted else 'rejected', image['image'], image['types'])
                    else:
                        accepted = True
                    if accepted:
                        urls = caa_url_fallback_list(config.setting['caa_image_size'], image['thumbnails'])
                        if not urls or is_pdf:
                            url = image['image']
                        else:
                            url = urls[0]
                        coverartimage = self.coverartimage_class(url, types=image['types'], is_front=image['front'], comment=image['comment'])
                        if urls and is_pdf:
                            thumbnail = self.coverartimage_thumbnail_class(url=urls[0], types=image['types'], is_front=image['front'], comment=image['comment'])
                            self.queue_put(thumbnail)
                            coverartimage.thumbnail = thumbnail
                            coverartimage.can_be_saved_to_tags = False
                        self.queue_put(coverartimage)
                        if config.setting['save_only_one_front_image'] and config.setting['save_images_to_files'] and image['front']:
                            break
            except (AttributeError, KeyError, TypeError) as e:
                self.error('CAA JSON error: %s' % e)
        self.next_in_queue()