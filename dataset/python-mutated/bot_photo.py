import os
from io import open
from tqdm import tqdm

def upload_photo(self, photo, caption=None, upload_id=None, from_video=False, options={}, user_tags=None, is_sidecar=False):
    if False:
        print('Hello World!')
    'Upload photo to Instagram\n\n    @param photo       Path to photo file (String)\n    @param caption     Media description (String)\n    @param upload_id   Unique upload_id (String). When None, then\n                       generate automatically\n    @param from_video  A flag that signals whether the photo is loaded from\n                       the video or by itself (Boolean, DEPRECATED: not used)\n    @param options     Object with difference options, e.g.\n                       configure_timeout, rename (Dict)\n                       Designed to reduce the number of function arguments!\n                       This is the simplest request object.\n    @param user_tags   Tag other users (List)\n                       usertags = [\n                         {"user_id": user_id, "position": [x, y]}\n                       ]\n    @param is_sidecar  An album element (Boolean)\n\n    @return            Object with state of uploading to Instagram (or False), Dict for is_sidecar\n    '
    self.small_delay()
    result = self.api.upload_photo(photo, caption, upload_id, from_video, options=options, user_tags=user_tags, is_sidecar=is_sidecar)
    if not result:
        self.logger.info("Photo '{}' is not uploaded.".format(photo))
        return False
    self.logger.info("Photo '{}' is uploaded.".format(photo))
    return result

def upload_album(self, photos, caption=None, upload_id=None, from_video=False, options={}, user_tags=None):
    if False:
        while True:
            i = 10
    'Upload album to Instagram\n\n    @param photos      List of paths to photo files (List of strings)\n    @param caption     Media description (String)\n    @param upload_id   Unique upload_id (String). When None, then\n                       generate automatically\n    @param from_video  A flag that signals whether the photo is loaded from\n                       the video or by itself (Boolean, DEPRECATED: not used)\n    @param options     Object with difference options, e.g.\n                       configure_timeout, rename (Dict)\n                       Designed to reduce the number of function arguments!\n                       This is the simplest request object.\n    @param user_tags\n\n    @return            Boolean\n    '
    self.small_delay()
    result = self.api.upload_album(photos, caption, upload_id, from_video, options=options, user_tags=user_tags)
    if not result:
        self.logger.info('Photos are not uploaded.')
        return False
    self.logger.info('Photo are uploaded.')
    return result

def download_photo(self, media_id, folder='photos', filename=None, save_description=False):
    if False:
        while True:
            i = 10
    self.small_delay()
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save_description:
        media = self.get_media_info(media_id)[0]
        caption = media['caption']['text'] if media['caption'] else ''
        username = media['user']['username']
        fname = os.path.join(folder, '{}_{}.txt'.format(username, media_id))
        with open(fname, encoding='utf8', mode='w') as f:
            f.write(caption)
    try:
        return self.api.download_photo(media_id, filename, False, folder)
    except Exception:
        self.logger.info('Media with `{}` is not downloaded.'.format(media_id))
        return False

def download_photos(self, medias, folder, save_description=False):
    if False:
        while True:
            i = 10
    broken_items = []
    if not medias:
        self.logger.info('Nothing to downloads.')
        return broken_items
    self.logger.info('Going to download {} medias.'.format(len(medias)))
    for media in tqdm(medias):
        if not self.download_photo(media, folder, save_description=save_description):
            self.error_delay()
            broken_items = medias[medias.index(media):]
    return broken_items