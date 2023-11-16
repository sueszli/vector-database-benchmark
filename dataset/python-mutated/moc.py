import os
import subprocess
from functools import partial
from libqtile.widget import base

class Moc(base.ThreadPoolText):
    """A simple MOC widget.

    Show the artist and album of now listening song and allow basic mouse
    control from the bar:

    - toggle pause (or play if stopped) on left click;
    - skip forward in playlist on scroll up;
    - skip backward in playlist on scroll down.

    MOC (http://moc.daper.net) should be installed.
    """
    defaults = [('play_color', '00ff00', 'Text colour when playing.'), ('noplay_color', 'cecece', 'Text colour when not playing.'), ('update_interval', 0.5, 'Update Time in seconds.')]

    def __init__(self, **config):
        if False:
            i = 10
            return i + 15
        base.ThreadPoolText.__init__(self, '', **config)
        self.add_defaults(Moc.defaults)
        self.status = ''
        self.local = None
        self.add_callbacks({'Button1': self.play, 'Button4': partial(subprocess.Popen, ['mocp', '-f']), 'Button5': partial(subprocess.Popen, ['mocp', '-r'])})

    def get_info(self):
        if False:
            i = 10
            return i + 15
        'Return a dictionary with info about the current MOC status.'
        try:
            output = self.call_process(['mocp', '-i'])
        except subprocess.CalledProcessError as err:
            output = err.output
        if output.startswith('State'):
            output = output.splitlines()
            info = {'State': '', 'File': '', 'SongTitle': '', 'Artist': '', 'Album': ''}
            for line in output:
                for data in info:
                    if data in line:
                        info[data] = line[len(data) + 2:].strip()
                        break
            return info

    def now_playing(self):
        if False:
            i = 10
            return i + 15
        'Return a string with the now playing info (Artist - Song Title).'
        info = self.get_info()
        now_playing = ''
        if info:
            status = info['State']
            if self.status != status:
                self.status = status
                if self.status == 'PLAY':
                    self.layout.colour = self.play_color
                else:
                    self.layout.colour = self.noplay_color
            title = info['SongTitle']
            artist = info['Artist']
            if title and artist:
                now_playing = '♫ {0} - {1}'.format(artist, title)
            elif title:
                now_playing = '♫ {0}'.format(title)
            else:
                basename = os.path.basename(info['File'])
                filename = os.path.splitext(basename)[0]
                now_playing = '♫ {0}'.format(filename)
            if self.status == 'STOP':
                now_playing = '♫'
        return now_playing

    def play(self):
        if False:
            for i in range(10):
                print('nop')
        'Play music if stopped, else toggle pause.'
        if self.status in ('PLAY', 'PAUSE'):
            subprocess.Popen(['mocp', '-G'])
        elif self.status == 'STOP':
            subprocess.Popen(['mocp', '-p'])

    def poll(self):
        if False:
            while True:
                i = 10
        'Poll content for the text box.'
        return self.now_playing()