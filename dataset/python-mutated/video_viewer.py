"""
This example demonstrates how static files can be served by making use
of a static file server.

If you intend to create a web application, note that using a static
server is a potential security risk. Use only when needed. Other options
that scale better for large websites are e.g. Nginx, Apache, or 3d party
services like Azure Storage or Amazon S3.

When exported, any links to local files wont work, but the remote links will.
"""
import os
from flexx import flx
from tornado.web import StaticFileHandler
dirname = os.path.expanduser('~/Videos')
videos = {}
for fname in os.listdir(dirname):
    if fname.endswith('.mp4'):
        videos[fname] = '/videos/' + fname
videos['bbb.mp4 (online)'] = 'http://www.w3schools.com/tags/mov_bbb.mp4'
videos['ice-age.mp4 (online)'] = 'https://dl.dropboxusercontent.com/u/1463853/ice%20age%204%20trailer.mp4'
tornado_app = flx.create_server().app
tornado_app.add_handlers('.*', [('/videos/(.*)', StaticFileHandler, {'path': dirname})])

class VideoViewer(flx.Widget):
    """ A simple videoviewer that displays a list of videos found on the
    server's computer, plus a few online videos. Note that not all videos
    may be playable in HTML5.
    """

    def init(self):
        if False:
            i = 10
            return i + 15
        with flx.HSplit():
            with flx.TreeWidget(max_selected=1, flex=1) as self.videolist:
                for name in sorted(videos):
                    flx.TreeItem(text=name)
            self.player = flx.VideoWidget(flex=5)

    @flx.reaction('videolist.children*.selected')
    def on_select(self, *events):
        if False:
            i = 10
            return i + 15
        for ev in events:
            if ev.source.selected:
                fname = ev.source.text
                self.player.set_source(videos[fname])
if __name__ == '__main__':
    m = flx.launch(VideoViewer)
    flx.run()