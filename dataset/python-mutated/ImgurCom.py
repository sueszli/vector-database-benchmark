import json
import re
from pyload.core.utils.purge import uniquify
from ..base.simple_decrypter import SimpleDecrypter

class ImgurCom(SimpleDecrypter):
    __name__ = 'ImgurCom'
    __type__ = 'decrypter'
    __version__ = '0.61'
    __status__ = 'testing'
    __pattern__ = 'https?://(?:www\\.|m\\.)?imgur\\.com/(a|gallery|)/?\\w{5,7}'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('folder_per_package', 'Default;Yes;No', 'Create folder for each package', 'Default'), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Imgur.com decrypter plugin'
    __license__ = 'GPLv3'
    __authors__ = [('nath_schwarz', 'nathan.notwhite@gmail.com'), ('nippey', 'matthias.nippert@gmail.com')]
    NAME_PATTERN = '(?P<N>.+?) - .*?Imgur'
    LINK_PATTERN = 'i\\.imgur\\.com/\\w{7}s?\\.(?:jpeg|jpg|png|gif|apng)'
    ' Imgur may only show the first 10 images of a gallery. The remaining bits may be found here: '
    GALLERY_JSON = 'http://imgur.com/ajaxalbums/getimages/{}/hit.json?all=true'

    def sanitize(self, name):
        if False:
            return 10
        '\n        Turn Imgur Gallery title into a safe Package and Folder name.\n        '
        keepcharacters = (' ', '\t', '.', '_')
        replacecharacters = (' ', '\t')
        return ''.join((c if c not in replacecharacters else '_' for c in name.strip() if c.isalnum() or c in keepcharacters)).strip('_')

    def get_ids_from_json(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check the embedded JSON and if needed the external JSON for more images.\n        '
        m = re.search('\\simage\\s+:\\s+({.*})', self.data)
        if m is not None:
            embedded_json = json.loads(m.group(1))
            gallery_id = embedded_json['hash']
            self.gallery_name = self.sanitize(self._('{}_{}').format(gallery_id, embedded_json['title']))
            self.total_num_images = int(embedded_json['num_images'])
            images = {e['hash']: e['ext'] for e in embedded_json['album_images']['images']}
            self.log_debug('Found {} of {} expected links in embedded JSON'.format(len(images), self.total_num_images))
            if len(images) < self.total_num_images:
                external_json = json.loads(self.load(self.GALLERY_JSON.format(gallery_id)))
                try:
                    images = {e['hash']: e['ext'] for e in external_json['data']['images']}
                    self.log_debug('Found {} of {} expected links in external JSON'.format(len(images), self.total_num_images))
                except (KeyError, TypeError):
                    self.log_debug('Could not extract links from external JSON')
            return images
        self.log_debug('Could not find embedded JSON')
        return {}

    def get_indirect_links(self, links_direct):
        if False:
            while True:
                i = 10
        "\n        Try to find a list of all images and add those we didn't find already.\n        "
        ids_direct = set((l for link in links_direct for l in re.findall('(\\w{7})', link)))
        ids_json = self.get_ids_from_json()
        ids_indirect = [id for id in ids_json.keys() if id not in ids_direct]
        if len(ids_indirect) == 0:
            return []
        return ['http://i.imgur.com/{}{}'.format(id, ids_json[id]) for id in ids_indirect]

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.gallery_name = None
        self.total_num_images = 0

    def get_links(self):
        if False:
            print('Hello World!')
        '\n        Extract embedded links from HTML // then check if there are further images which\n        will be lazy-loaded.\n        '

        def f(url):
            if False:
                while True:
                    i = 10
            return 'http://' + re.sub('(\\w{7})s\\.', '\\1.', url)
        direct_links = uniquify((f(x) for x in re.findall(self.LINK_PATTERN, self.data)))
        try:
            indirect_links = self.get_indirect_links(direct_links)
            self.log_debug(f'Found {len(indirect_links)} additional links')
        except (TypeError, KeyError, ValueError) as exc:
            self.log_error(self._('Processing of additional links unsuccessful - {}: {}').format(type(exc).__name__, exc))
            indirect_links = []
        num_images_found = len(direct_links) + len(indirect_links)
        if num_images_found < self.total_num_images:
            self.log_error(self._('Could not save all images of this gallery: {}/{}').format(num_images_found, self.total_num_images))
        if self.gallery_name:
            self.packages.append((self.gallery_name, direct_links + indirect_links, self.gallery_name))
            return []
        else:
            return direct_links + indirect_links