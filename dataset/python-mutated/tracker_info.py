"""
Keeping track of information about a tracker.
"""

class TrackerInfo:
    """
    This class keeps track of info about a tracker. This info is used when a request to a tracker is performed.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.infohashes = {}

    def add_info_about_infohash(self, infohash, seeders, leechers, downloaded=0):
        if False:
            i = 10
            return i + 15
        '\n        Add information about an infohash to our tracker info.\n        '
        self.infohashes[infohash] = {'seeders': seeders, 'leechers': leechers, 'downloaded': downloaded}

    def get_info_about_infohash(self, infohash):
        if False:
            return 10
        '\n        Returns information about an infohash, None if this infohash is not in our info.\n        '
        if infohash not in self.infohashes:
            return None
        return self.infohashes[infohash]

    def has_info_about_infohash(self, infohash):
        if False:
            i = 10
            return i + 15
        '\n        Return True if we have information about a specified infohash\n        '
        return infohash in self.infohashes