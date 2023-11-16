from tqdm import tqdm

def delete_media(self, media_id):
    if False:
        return 10
    self.small_delay()
    media = self.get_media_info(media_id)
    media = media[0] if isinstance(media, list) else media
    if self.api.delete_media(media):
        return True
    self.logger.info('Media with {} is not {}.'.format(media.get('id'), 'deleted'))
    return False

def delete_medias(self, medias):
    if False:
        return 10
    broken_items = []
    if not medias:
        self.logger.info('Nothing to delete.')
        return broken_items
    self.logger.info('Going to delete %d medias.' % len(medias))
    for media in tqdm(medias):
        if not self.delete_media(media):
            self.error_delay()
            broken_items = medias[medias.index(media):]
            break
    self.logger.info('DONE: Total deleted %d medias.' % (len(medias) - len(broken_items)))
    return broken_items

def delete_comment(self, media_id, comment_id):
    if False:
        i = 10
        return i + 15
    if self.api.delete_comment(media_id, comment_id):
        self.small_delay()
        return True
    self.logger.info('Comment with {} in media {} is not deleted.'.format(comment_id, media_id))
    return False