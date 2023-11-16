from urllib.parse import quote_plus
from plexapi import utils
from plexapi.base import PlexObject
from plexapi.exceptions import BadRequest

class PlayQueue(PlexObject):
    """Control a PlayQueue.

    Attributes:
        TAG (str): 'PlayQueue'
        TYPE (str): 'playqueue'
        identifier (str): com.plexapp.plugins.library
        items (list): List of :class:`~plexapi.base.Playable` or :class:`~plexapi.playlist.Playlist`
        mediaTagPrefix (str): Fx /system/bundle/media/flags/
        mediaTagVersion (int): Fx 1485957738
        playQueueID (int): ID of the PlayQueue.
        playQueueLastAddedItemID (int):
            Defines where the "Up Next" region starts. Empty unless PlayQueue is modified after creation.
        playQueueSelectedItemID (int): The queue item ID of the currently selected item.
        playQueueSelectedItemOffset (int):
            The offset of the selected item in the PlayQueue, from the beginning of the queue.
        playQueueSelectedMetadataItemID (int): ID of the currently selected item, matches ratingKey.
        playQueueShuffled (bool): True if shuffled.
        playQueueSourceURI (str): Original URI used to create the PlayQueue.
        playQueueTotalCount (int): How many items in the PlayQueue.
        playQueueVersion (int): Version of the PlayQueue. Increments every time a change is made to the PlayQueue.
        selectedItem (:class:`~plexapi.base.Playable`): Media object for the currently selected item.
        _server (:class:`~plexapi.server.PlexServer`): PlexServer associated with the PlayQueue.
        size (int): Alias for playQueueTotalCount.
    """
    TAG = 'PlayQueue'
    TYPE = 'playqueue'

    def _loadData(self, data):
        if False:
            while True:
                i = 10
        self._data = data
        self.identifier = data.attrib.get('identifier')
        self.mediaTagPrefix = data.attrib.get('mediaTagPrefix')
        self.mediaTagVersion = utils.cast(int, data.attrib.get('mediaTagVersion'))
        self.playQueueID = utils.cast(int, data.attrib.get('playQueueID'))
        self.playQueueLastAddedItemID = utils.cast(int, data.attrib.get('playQueueLastAddedItemID'))
        self.playQueueSelectedItemID = utils.cast(int, data.attrib.get('playQueueSelectedItemID'))
        self.playQueueSelectedItemOffset = utils.cast(int, data.attrib.get('playQueueSelectedItemOffset'))
        self.playQueueSelectedMetadataItemID = utils.cast(int, data.attrib.get('playQueueSelectedMetadataItemID'))
        self.playQueueShuffled = utils.cast(bool, data.attrib.get('playQueueShuffled', 0))
        self.playQueueSourceURI = data.attrib.get('playQueueSourceURI')
        self.playQueueTotalCount = utils.cast(int, data.attrib.get('playQueueTotalCount'))
        self.playQueueVersion = utils.cast(int, data.attrib.get('playQueueVersion'))
        self.size = utils.cast(int, data.attrib.get('size', 0))
        self.items = self.findItems(data)
        self.selectedItem = self[self.playQueueSelectedItemOffset]

    def __getitem__(self, key):
        if False:
            return 10
        if not self.items:
            return None
        return self.items[key]

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.playQueueTotalCount

    def __iter__(self):
        if False:
            while True:
                i = 10
        yield from self.items

    def __contains__(self, media):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if the PlayQueue contains the provided media item.'
        return any((x.playQueueItemID == media.playQueueItemID for x in self.items))

    def getQueueItem(self, item):
        if False:
            print('Hello World!')
        '\n        Accepts a media item and returns a similar object from this PlayQueue.\n        Useful for looking up playQueueItemIDs using items obtained from the Library.\n        '
        matches = [x for x in self.items if x == item]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise BadRequest(f'{item} occurs multiple times in this PlayQueue, provide exact item')
        else:
            raise BadRequest(f'{item} not valid for this PlayQueue')

    @classmethod
    def get(cls, server, playQueueID, own=False, center=None, window=50, includeBefore=True, includeAfter=True):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve an existing :class:`~plexapi.playqueue.PlayQueue` by identifier.\n\n        Parameters:\n            server (:class:`~plexapi.server.PlexServer`): Server you are connected to.\n            playQueueID (int): Identifier of an existing PlayQueue.\n            own (bool, optional): If server should transfer ownership.\n            center (int, optional): The playQueueItemID of the center of the window. Does not change selectedItem.\n            window (int, optional): Number of items to return from each side of the center item.\n            includeBefore (bool, optional):\n                Include items before the center, defaults True. Does not include center if False.\n            includeAfter (bool, optional):\n                Include items after the center, defaults True. Does not include center if False.\n        '
        args = {'own': utils.cast(int, own), 'window': window, 'includeBefore': utils.cast(int, includeBefore), 'includeAfter': utils.cast(int, includeAfter)}
        if center:
            args['center'] = center
        path = f'/playQueues/{playQueueID}{utils.joinArgs(args)}'
        data = server.query(path, method=server._session.get)
        c = cls(server, data, initpath=path)
        c._server = server
        return c

    @classmethod
    def create(cls, server, items, startItem=None, shuffle=0, repeat=0, includeChapters=1, includeRelated=1, continuous=0):
        if False:
            i = 10
            return i + 15
        'Create and return a new :class:`~plexapi.playqueue.PlayQueue`.\n\n        Parameters:\n            server (:class:`~plexapi.server.PlexServer`): Server you are connected to.\n            items (:class:`~plexapi.base.PlexPartialObject`):\n                A media item or a list of media items.\n            startItem (:class:`~plexapi.base.Playable`, optional):\n                Media item in the PlayQueue where playback should begin.\n            shuffle (int, optional): Start the playqueue shuffled.\n            repeat (int, optional): Start the playqueue shuffled.\n            includeChapters (int, optional): include Chapters.\n            includeRelated (int, optional): include Related.\n            continuous (int, optional): include additional items after the initial item.\n                For a show this would be the next episodes, for a movie it does nothing.\n        '
        args = {'includeChapters': includeChapters, 'includeRelated': includeRelated, 'repeat': repeat, 'shuffle': shuffle, 'continuous': continuous}
        if isinstance(items, list):
            item_keys = ','.join((str(x.ratingKey) for x in items))
            uri_args = quote_plus(f'/library/metadata/{item_keys}')
            args['uri'] = f'library:///directory/{uri_args}'
            args['type'] = items[0].listType
        else:
            if items.type == 'playlist':
                args['type'] = items.playlistType
                args['playlistID'] = items.ratingKey
            else:
                args['type'] = items.listType
            args['uri'] = f'server://{server.machineIdentifier}/{server.library.identifier}{items.key}'
        if startItem:
            args['key'] = startItem.key
        path = f'/playQueues{utils.joinArgs(args)}'
        data = server.query(path, method=server._session.post)
        c = cls(server, data, initpath=path)
        c._server = server
        return c

    @classmethod
    def fromStationKey(cls, server, key):
        if False:
            while True:
                i = 10
        'Create and return a new :class:`~plexapi.playqueue.PlayQueue`.\n\n        This is a convenience method to create a `PlayQueue` for\n        radio stations when only the `key` string is available.\n\n        Parameters:\n            server (:class:`~plexapi.server.PlexServer`): Server you are connected to.\n            key (str): A station key as provided by :func:`~plexapi.library.LibrarySection.hubs()`\n                or :func:`~plexapi.audio.Artist.station()`\n\n        Example:\n\n            .. code-block:: python\n\n                from plexapi.playqueue import PlayQueue\n                music = server.library.section("Music")\n                artist = music.get("Artist Name")\n                station = artist.station()\n                key = station.key  # "/library/metadata/12855/station/8bd39616-dbdb-459e-b8da-f46d0b170af4?type=10"\n                pq = PlayQueue.fromStationKey(server, key)\n                client = server.clients()[0]\n                client.playMedia(pq)\n        '
        args = {'type': 'audio', 'uri': f'server://{server.machineIdentifier}/{server.library.identifier}{key}'}
        path = f'/playQueues{utils.joinArgs(args)}'
        data = server.query(path, method=server._session.post)
        c = cls(server, data, initpath=path)
        c._server = server
        return c

    def addItem(self, item, playNext=False, refresh=True):
        if False:
            i = 10
            return i + 15
        '\n        Append the provided item to the "Up Next" section of the PlayQueue.\n        Items can only be added to the section immediately following the current playing item.\n\n        Parameters:\n            item (:class:`~plexapi.base.Playable` or :class:`~plexapi.playlist.Playlist`): Single media item or Playlist.\n            playNext (bool, optional): If True, add this item to the front of the "Up Next" section.\n                If False, the item will be appended to the end of the "Up Next" section.\n                Only has an effect if an item has already been added to the "Up Next" section.\n                See https://support.plex.tv/articles/202188298-play-queues/ for more details.\n            refresh (bool, optional): Refresh the PlayQueue from the server before updating.\n        '
        if refresh:
            self.refresh()
        args = {}
        if item.type == 'playlist':
            args['playlistID'] = item.ratingKey
        else:
            uuid = item.section().uuid
            args['uri'] = f'library://{uuid}/item{item.key}'
        if playNext:
            args['next'] = 1
        path = f'/playQueues/{self.playQueueID}{utils.joinArgs(args)}'
        data = self._server.query(path, method=self._server._session.put)
        self._loadData(data)
        return self

    def moveItem(self, item, after=None, refresh=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Moves an item to the beginning of the PlayQueue.  If `after` is provided,\n        the item will be placed immediately after the specified item.\n\n        Parameters:\n            item (:class:`~plexapi.base.Playable`): An existing item in the PlayQueue to move.\n            afterItemID (:class:`~plexapi.base.Playable`, optional): A different item in the PlayQueue.\n                If provided, `item` will be placed in the PlayQueue after this item.\n            refresh (bool, optional): Refresh the PlayQueue from the server before updating.\n        '
        args = {}
        if refresh:
            self.refresh()
        if item not in self:
            item = self.getQueueItem(item)
        if after:
            if after not in self:
                after = self.getQueueItem(after)
            args['after'] = after.playQueueItemID
        path = f'/playQueues/{self.playQueueID}/items/{item.playQueueItemID}/move{utils.joinArgs(args)}'
        data = self._server.query(path, method=self._server._session.put)
        self._loadData(data)
        return self

    def removeItem(self, item, refresh=True):
        if False:
            i = 10
            return i + 15
        'Remove an item from the PlayQueue.\n\n        Parameters:\n            item (:class:`~plexapi.base.Playable`): An existing item in the PlayQueue to move.\n            refresh (bool, optional): Refresh the PlayQueue from the server before updating.\n        '
        if refresh:
            self.refresh()
        if item not in self:
            item = self.getQueueItem(item)
        path = f'/playQueues/{self.playQueueID}/items/{item.playQueueItemID}'
        data = self._server.query(path, method=self._server._session.delete)
        self._loadData(data)
        return self

    def clear(self):
        if False:
            print('Hello World!')
        'Remove all items from the PlayQueue.'
        path = f'/playQueues/{self.playQueueID}/items'
        data = self._server.query(path, method=self._server._session.delete)
        self._loadData(data)
        return self

    def refresh(self):
        if False:
            i = 10
            return i + 15
        'Refresh the PlayQueue from the Plex server.'
        path = f'/playQueues/{self.playQueueID}'
        data = self._server.query(path, method=self._server._session.get)
        self._loadData(data)
        return self