import heapq

class VideoSharingPlatform(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__avails = []
        self.__videos = []
        self.__likes = []
        self.__dislikes = []
        self.__views = []

    def upload(self, video):
        if False:
            print('Hello World!')
        '\n        :type video: str\n        :rtype: int\n        '
        if self.__avails:
            i = heapq.heappop(self.__avails)
        else:
            i = len(self.__videos)
            self.__videos.append(None)
            self.__likes.append(0)
            self.__dislikes.append(0)
            self.__views.append(0)
        self.__videos[i] = video
        return i

    def remove(self, videoId):
        if False:
            print('Hello World!')
        '\n        :type videoId: int\n        :rtype: None\n        '
        if videoId >= len(self.__videos) or not self.__videos[videoId]:
            return
        heapq.heappush(self.__avails, videoId)
        self.__videos[videoId] = None
        self.__likes[videoId] = self.__dislikes[videoId] = self.__views[videoId] = 0

    def watch(self, videoId, startMinute, endMinute):
        if False:
            return 10
        '\n        :type videoId: int\n        :type startMinute: int\n        :type endMinute: int\n        :rtype: str\n        '
        if videoId >= len(self.__videos) or not self.__videos[videoId]:
            return '-1'
        self.__views[videoId] += 1
        return self.__videos[videoId][startMinute:endMinute + 1]

    def like(self, videoId):
        if False:
            print('Hello World!')
        '\n        :type videoId: int\n        :rtype: None\n        '
        if videoId >= len(self.__videos) or not self.__videos[videoId]:
            return
        self.__likes[videoId] += 1

    def dislike(self, videoId):
        if False:
            print('Hello World!')
        '\n        :type videoId: int\n        :rtype: None\n        '
        if videoId >= len(self.__videos) or not self.__videos[videoId]:
            return
        self.__dislikes[videoId] += 1

    def getLikesAndDislikes(self, videoId):
        if False:
            while True:
                i = 10
        '\n        :type videoId: int\n        :rtype: List[int]\n        '
        if videoId >= len(self.__videos) or not self.__videos[videoId]:
            return [-1]
        return [self.__likes[videoId], self.__dislikes[videoId]]

    def getViews(self, videoId):
        if False:
            i = 10
            return i + 15
        '\n        :type videoId: int\n        :rtype: int\n        '
        if videoId >= len(self.__videos) or not self.__videos[videoId]:
            return -1
        return self.__views[videoId]