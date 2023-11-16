import kivy
kivy.require('1.2.0')
from sys import argv
from os.path import dirname, join
from kivy.app import App
from kivy.uix.videoplayer import VideoPlayer

class VideoPlayerApp(App):

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        if len(argv) > 1:
            filename = argv[1]
        else:
            curdir = dirname(__file__)
            filename = join(curdir, 'cityCC0.mpg')
        return VideoPlayer(source=filename, state='play')
if __name__ == '__main__':
    VideoPlayerApp().run()