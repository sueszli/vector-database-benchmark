from Foundation import NSRunLoop, NSMutableDictionary, NSObject
from MediaPlayer import MPRemoteCommandCenter, MPNowPlayingInfoCenter
from MediaPlayer import MPMediaItemPropertyTitle, MPMediaItemPropertyArtist, MPMusicPlaybackState, MPMusicPlaybackStatePlaying, MPMusicPlaybackStatePaused, MPNowPlayingInfoPropertyElapsedPlaybackTime, MPMediaItemPropertyPlaybackDuration

class NowPlaying:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.cmd_center = MPRemoteCommandCenter.sharedCommandCenter()
        self.info_center = MPNowPlayingInfoCenter.defaultCenter()
        cmds = [self.cmd_center.togglePlayPauseCommand(), self.cmd_center.playCommand(), self.cmd_center.pauseCommand()]
        for cmd in cmds:
            cmd.addTargetWithHandler_(self._create_handler(cmd))
        self.update_info()

    def update_info(self):
        if False:
            for i in range(10):
                print('nop')
        nowplaying_info = NSMutableDictionary.dictionary()
        nowplaying_info[MPMediaItemPropertyTitle] = 'title'
        nowplaying_info[MPMediaItemPropertyArtist] = 'artist'
        nowplaying_info[MPNowPlayingInfoPropertyElapsedPlaybackTime] = 0
        nowplaying_info[MPMediaItemPropertyPlaybackDuration] = 100
        self.info_center.setNowPlayingInfo_(nowplaying_info)
        self.info_center.setPlaybackState_(MPMusicPlaybackStatePlaying)

    def _create_handler(self, cmd):
        if False:
            while True:
                i = 10

        def handle(event):
            if False:
                while True:
                    i = 10
            if event.command() == self.cmd_center.pauseCommand():
                self.info_center.setPlaybackState_(MPMusicPlaybackStatePaused)
            elif event.command() == self.cmd_center.playCommand():
                self.info_center.setPlaybackState_(MPMusicPlaybackStatePlaying)
            return 0
        return handle

def runloop():
    if False:
        print('Hello World!')
    "\n    HELP: This function can't be called in non-main thread.\n    "
    nowplaying = NowPlaying()
    NSRunLoop.mainRunLoop().run()
runloop()