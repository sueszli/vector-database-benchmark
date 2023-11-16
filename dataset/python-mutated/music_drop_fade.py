""" pygame.examples.music_drop_fade
Fade in and play music from a list while observing several events

Adds music files to a playlist whenever played by one of the following methods
Music files passed from the commandline are played
Music files and filenames are played when drag and dropped onto the pygame window
Polls the clipboard and plays music files if it finds one there

Keyboard Controls:
* Press space or enter to pause music playback
* Press up or down to change the music volume
* Press left or right to seek 5 seconds into the track
* Press escape to quit
* Press any other button to skip to the next music file in the list
"""
from typing import List
import pygame as pg
import os, sys
VOLUME_CHANGE_AMOUNT = 0.02

def add_file(filename):
    if False:
        while True:
            i = 10
    '\n    This function will check if filename exists and is a music file\n    If it is the file will be added to a list of music files(even if already there)\n    Type checking is by the extension of the file, not by its contents\n    We can only discover if the file is valid when we mixer.music.load() it later\n\n    It looks in the file directory and its data subdirectory\n    '
    if filename.rpartition('.')[2].lower() not in music_file_types:
        print(f'{filename} not added to file list')
        print('only these files types are allowed: ', music_file_types)
        return False
    elif os.path.exists(filename):
        music_file_list.append(filename)
    elif os.path.exists(os.path.join(main_dir, filename)):
        music_file_list.append(os.path.join(main_dir, filename))
    elif os.path.exists(os.path.join(data_dir, filename)):
        music_file_list.append(os.path.join(data_dir, filename))
    else:
        print('file not found')
        return False
    print(f'{filename} added to file list')
    return True

def play_file(filename):
    if False:
        return 10
    '\n    This function will call add_file and play it if successful\n    The music will fade in during the first 4 seconds\n    set_endevent is used to post a MUSIC_DONE event when the song finishes\n    The main loop will call play_next() when the MUSIC_DONE event is received\n    '
    global starting_pos
    if add_file(filename):
        try:
            pg.mixer.music.load(music_file_list[-1])
        except pg.error as e:
            print(e)
            if filename in music_file_list:
                music_file_list.remove(filename)
                print(f'{filename} removed from file list')
            return
        pg.mixer.music.play(fade_ms=4000)
        pg.mixer.music.set_volume(volume)
        if filename.rpartition('.')[2].lower() in music_can_seek:
            print('file supports seeking')
            starting_pos = 0
        else:
            print('file does not support seeking')
            starting_pos = -1
        pg.mixer.music.set_endevent(MUSIC_DONE)

def play_next():
    if False:
        while True:
            i = 10
    '\n    This function will play the next song in music_file_list\n    It uses pop(0) to get the next song and then appends it to the end of the list\n    The song will fade in during the first 4 seconds\n    '
    global starting_pos
    if len(music_file_list) > 1:
        nxt = music_file_list.pop(0)
        try:
            pg.mixer.music.load(nxt)
        except pg.error as e:
            print(e)
            print(f'{nxt} removed from file list')
        music_file_list.append(nxt)
        print('starting next song: ', nxt)
    else:
        nxt = music_file_list[0]
    pg.mixer.music.play(fade_ms=4000)
    pg.mixer.music.set_volume(volume)
    pg.mixer.music.set_endevent(MUSIC_DONE)
    if nxt.rpartition('.')[2].lower() in music_can_seek:
        starting_pos = 0
    else:
        starting_pos = -1

def draw_text_line(text, y=0):
    if False:
        i = 10
        return i + 15
    "\n    Draws a line of text onto the display surface\n    The text will be centered horizontally at the given y position\n    The text's height is added to y and returned to the caller\n    "
    screen = pg.display.get_surface()
    surf = font.render(text, 1, (255, 255, 255))
    y += surf.get_height()
    x = (screen.get_width() - surf.get_width()) / 2
    screen.blit(surf, (x, y))
    return y

def change_music_position(amount):
    if False:
        for i in range(10):
            print('nop')
    '\n    Changes current playback position by amount seconds.\n    This only works with OGG and MP3 files.\n    music.get_pos() returns how many milliseconds the song has played, not\n    the current position in the file. We must track the starting position\n    ourselves. music.set_pos() will set the position in seconds.\n    '
    global starting_pos
    if starting_pos >= 0:
        played_for = pg.mixer.music.get_pos() / 1000.0
        old_pos = starting_pos + played_for
        starting_pos = old_pos + amount
        pg.mixer.music.play(start=starting_pos)
        print(f'jumped from {old_pos} to {starting_pos}')
MUSIC_DONE = pg.event.custom_type()
main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, 'data')
starting_pos = 0
volume = 0.75
music_file_list: List[str] = []
music_file_types = ('mp3', 'ogg', 'mid', 'mod', 'it', 'xm', 'wav')
music_can_seek = ('mp3', 'ogg', 'mod', 'it', 'xm')

def main():
    if False:
        i = 10
        return i + 15
    global font
    global volume, starting_pos
    running = True
    paused = False
    change_volume = 0
    pg.init()
    pg.display.set_mode((640, 480))
    font = pg.font.SysFont('Arial', 24)
    clock = pg.time.Clock()
    pg.scrap.init()
    pg.SCRAP_TEXT = pg.scrap.get_types()[0]
    scrap_get = pg.scrap.get(pg.SCRAP_TEXT)
    clipped = '' if scrap_get is None else scrap_get.decode('UTF-8')
    for arg in sys.argv[1:]:
        add_file(arg)
    play_file('house_lo.ogg')
    y = draw_text_line('Drop music files or path names onto this window', 20)
    y = draw_text_line('Copy file names into the clipboard', y)
    y = draw_text_line('Or feed them from the command line', y)
    y = draw_text_line("If it's music it will play!", y)
    y = draw_text_line('SPACE to pause or UP/DOWN to change volume', y)
    y = draw_text_line('LEFT and RIGHT will skip around the track', y)
    draw_text_line('Other keys will start the next track', y)
    '\n    This is the main loop\n    It will respond to drag and drop, clipboard changes, and key presses\n    '
    while running:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.DROPTEXT:
                play_file(ev.text)
            elif ev.type == pg.DROPFILE:
                play_file(ev.file)
            elif ev.type == MUSIC_DONE:
                play_next()
            elif ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    running = False
                elif ev.key in (pg.K_SPACE, pg.K_RETURN):
                    if paused:
                        pg.mixer.music.unpause()
                        paused = False
                    else:
                        pg.mixer.music.pause()
                        paused = True
                elif ev.key == pg.K_UP:
                    change_volume = VOLUME_CHANGE_AMOUNT
                elif ev.key == pg.K_DOWN:
                    change_volume = -VOLUME_CHANGE_AMOUNT
                elif ev.key == pg.K_RIGHT:
                    change_music_position(+5)
                elif ev.key == pg.K_LEFT:
                    change_music_position(-5)
                else:
                    play_next()
            elif ev.type == pg.KEYUP:
                if ev.key in (pg.K_UP, pg.K_DOWN):
                    change_volume = 0
        if change_volume:
            volume += change_volume
            volume = min(max(0, volume), 1)
            pg.mixer.music.set_volume(volume)
            print('volume:', volume)
        scrap_get = pg.scrap.get(pg.SCRAP_TEXT)
        new_text = '' if scrap_get is None else scrap_get.decode('UTF-8')
        if new_text != clipped:
            clipped = new_text
            play_file(clipped)
        pg.display.flip()
        clock.tick(9)
    pg.quit()
if __name__ == '__main__':
    main()