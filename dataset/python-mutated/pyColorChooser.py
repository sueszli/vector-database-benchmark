"""Color chooser implementing (almost) the tkColorColor interface
"""
import os
import Main
import ColorDB

class Chooser:
    """Ask for a color"""

    def __init__(self, master=None, databasefile=None, initfile=None, ignore=None, wantspec=None):
        if False:
            return 10
        self.__master = master
        self.__databasefile = databasefile
        self.__initfile = initfile or os.path.expanduser('~/.pynche')
        self.__ignore = ignore
        self.__pw = None
        self.__wantspec = wantspec

    def show(self, color, options):
        if False:
            for i in range(10):
                print('nop')
        self.__wantspec = options.get('wantspec', self.__wantspec)
        dbfile = options.get('databasefile', self.__databasefile)
        colordb = None
        if dbfile != self.__databasefile:
            colordb = ColorDB.get_colordb(dbfile)
        if not self.__master:
            from tkinter import Tk
            self.__master = Tk()
        if not self.__pw:
            (self.__pw, self.__sb) = Main.build(master=self.__master, initfile=self.__initfile, ignore=self.__ignore)
        else:
            self.__pw.deiconify()
        if colordb:
            self.__sb.set_colordb(colordb)
        else:
            colordb = self.__sb.colordb()
        if color:
            (r, g, b) = Main.initial_color(color, colordb)
            self.__sb.update_views(r, g, b)
        self.__sb.canceled(0)
        Main.run(self.__pw, self.__sb)
        rgbtuple = self.__sb.current_rgb()
        self.__pw.withdraw()
        if self.__sb.canceled_p():
            return (None, None)
        name = None
        if not self.__wantspec:
            try:
                name = colordb.find_byrgb(rgbtuple)[0]
            except ColorDB.BadColor:
                pass
        if name is None:
            name = ColorDB.triplet_to_rrggbb(rgbtuple)
        return (rgbtuple, name)

    def save(self):
        if False:
            i = 10
            return i + 15
        if self.__sb:
            self.__sb.save_views()
_chooser = None

def askcolor(color=None, **options):
    if False:
        return 10
    'Ask for a color'
    global _chooser
    if not _chooser:
        _chooser = Chooser(**options)
    return _chooser.show(color, options)

def save():
    if False:
        while True:
            i = 10
    global _chooser
    if _chooser:
        _chooser.save()
if __name__ == '__main__':
    from tkinter import *

    class Tester:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.__root = tk = Tk()
            b = Button(tk, text='Choose Color...', command=self.__choose)
            b.pack()
            self.__l = Label(tk)
            self.__l.pack()
            q = Button(tk, text='Quit', command=self.__quit)
            q.pack()

        def __choose(self, event=None):
            if False:
                return 10
            (rgb, name) = askcolor(master=self.__root)
            if rgb is None:
                text = 'You hit CANCEL!'
            else:
                (r, g, b) = rgb
                text = 'You picked %s (%3d/%3d/%3d)' % (name, r, g, b)
            self.__l.configure(text=text)

        def __quit(self, event=None):
            if False:
                print('Hello World!')
            self.__root.quit()

        def run(self):
            if False:
                print('Hello World!')
            self.__root.mainloop()
    t = Tester()
    t.run()