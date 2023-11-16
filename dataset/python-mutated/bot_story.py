def download_stories(self, username):
    if False:
        for i in range(10):
            print('nop')
    user_id = self.get_user_id_from_username(username)
    (list_image, list_video) = self.get_user_stories(user_id)
    if list_image == [] and list_video == []:
        self.logger.error("Make sure that '{}' is NOT private and that posted some stories".format(username))
        return False
    self.logger.info('Downloading stories...')
    for story_url in list_image:
        filename = story_url.split('/')[-1].split('.')[0] + '.jpg'
        self.api.download_story(filename, story_url, username)
    for story_url in list_video:
        filename = story_url.split('/')[-1].split('.')[0] + '.mp4'
        self.api.download_story(filename, story_url, username)

def upload_story_photo(self, photo, upload_id=None):
    if False:
        print('Hello World!')
    self.small_delay()
    if self.api.upload_story_photo(photo, upload_id):
        self.logger.info("Photo '{}' is uploaded as Story.".format(photo))
        return True
    self.logger.info("Photo '{}' is not uploaded.".format(photo))
    return False

def watch_users_reels(self, user_ids, max_users=100):
    if False:
        i = 10
        return i + 15
    "\n        user_ids - the list of user_id to get their stories\n        max_users - max amount of users to get stories from.\n\n        It seems like Instagram doesn't allow to get stories\n        from more that 100 users at once.\n    "
    if not isinstance(user_ids, list):
        user_ids = [user_ids]
    reels = self.api.get_users_reel(user_ids[:max_users])
    if isinstance(reels, list):
        return False
    reels = {k: v for (k, v) in reels.items() if 'items' in v and len(v['items']) > 0}
    unseen_reels = []
    for (_, reels_data) in reels.items():
        last_reel_seen_at = reels_data['seen'] if 'seen' in reels_data else 0
        unseen_reels.extend([r for r in reels_data['items'] if r['taken_at'] > last_reel_seen_at])
    if self.api.see_reels(unseen_reels):
        self.total['stories_viewed'] += len(unseen_reels)
        return True
    return False