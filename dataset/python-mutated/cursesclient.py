"""
This is an example of integrating curses with the twisted underlying
select loop. Most of what is in this is insignificant -- the main piece
of interest is the 'CursesStdIO' class.

This class acts as file-descriptor 0, and is scheduled with the twisted
select loop via reactor.addReader (once the curses class extends it
of course). When there is input waiting doRead is called, and any
input-oriented curses calls (ie. getch()) should be executed within this
block.

Remember to call nodelay(1) in curses, to make getch() non-blocking.

To run the script::

    $ python cursesclient.py

"""
import curses
import curses.wrapper
from twisted.internet import reactor
from twisted.internet.protocol import ClientFactory
from twisted.words.protocols.irc import IRCClient

class TextTooLongError(Exception):
    pass

class CursesStdIO:
    """fake fd to be registered as a reader with the twisted reactor.
    Curses classes needing input should extend this"""

    def fileno(self):
        if False:
            print('Hello World!')
        'We want to select on FD 0'
        return 0

    def doRead(self):
        if False:
            i = 10
            return i + 15
        'called when input is ready'

    def logPrefix(self):
        if False:
            while True:
                i = 10
        return 'CursesClient'

class IRC(IRCClient):
    """A protocol object for IRC"""
    nickname = 'testcurses'

    def __init__(self, screenObj):
        if False:
            return 10
        self.screenObj = screenObj
        self.screenObj.irc = self

    def lineReceived(self, line):
        if False:
            i = 10
            return i + 15
        'When receiving a line, add it to the output buffer'
        self.screenObj.addLine(line)

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        IRCClient.connectionMade(self)
        self.screenObj.addLine('* CONNECTED')

    def clientConnectionLost(self, connection, reason):
        if False:
            for i in range(10):
                print('nop')
        pass

class IRCFactory(ClientFactory):
    """
    Factory used for creating IRC protocol objects
    """
    protocol = IRC

    def __init__(self, screenObj):
        if False:
            while True:
                i = 10
        self.irc = self.protocol(screenObj)

    def buildProtocol(self, addr=None):
        if False:
            print('Hello World!')
        return self.irc

    def clientConnectionLost(self, conn, reason):
        if False:
            i = 10
            return i + 15
        pass

class Screen(CursesStdIO):

    def __init__(self, stdscr):
        if False:
            print('Hello World!')
        self.timer = 0
        self.statusText = 'TEST CURSES APP -'
        self.searchText = ''
        self.stdscr = stdscr
        self.stdscr.nodelay(1)
        curses.cbreak()
        self.stdscr.keypad(1)
        curses.curs_set(0)
        (self.rows, self.cols) = self.stdscr.getmaxyx()
        self.lines = []
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
        self.paintStatus(self.statusText)

    def connectionLost(self, reason):
        if False:
            while True:
                i = 10
        self.close()

    def addLine(self, text):
        if False:
            return 10
        'add a line to the internal list of lines'
        self.lines.append(text)
        self.redisplayLines()

    def redisplayLines(self):
        if False:
            print('Hello World!')
        'method for redisplaying lines\n        based on internal list of lines'
        self.stdscr.clear()
        self.paintStatus(self.statusText)
        i = 0
        index = len(self.lines) - 1
        while i < self.rows - 3 and index >= 0:
            self.stdscr.addstr(self.rows - 3 - i, 0, self.lines[index], curses.color_pair(2))
            i = i + 1
            index = index - 1
        self.stdscr.refresh()

    def paintStatus(self, text):
        if False:
            return 10
        if len(text) > self.cols:
            raise TextTooLongError
        self.stdscr.addstr(self.rows - 2, 0, text + ' ' * (self.cols - len(text)), curses.color_pair(1))
        self.stdscr.move(self.rows - 1, self.cols - 1)

    def doRead(self):
        if False:
            for i in range(10):
                print('nop')
        'Input is ready!'
        curses.noecho()
        self.timer = self.timer + 1
        c = self.stdscr.getch()
        if c == curses.KEY_BACKSPACE:
            self.searchText = self.searchText[:-1]
        elif c == curses.KEY_ENTER or c == 10:
            self.addLine(self.searchText)
            try:
                self.irc.sendLine(self.searchText)
            except BaseException:
                pass
            self.stdscr.refresh()
            self.searchText = ''
        else:
            if len(self.searchText) == self.cols - 2:
                return
            self.searchText = self.searchText + chr(c)
        self.stdscr.addstr(self.rows - 1, 0, self.searchText + ' ' * (self.cols - len(self.searchText) - 2))
        self.stdscr.move(self.rows - 1, len(self.searchText))
        self.paintStatus(self.statusText + ' %d' % len(self.searchText))
        self.stdscr.refresh()

    def close(self):
        if False:
            i = 10
            return i + 15
        'clean up'
        curses.nocbreak()
        self.stdscr.keypad(0)
        curses.echo()
        curses.endwin()
if __name__ == '__main__':
    stdscr = curses.initscr()
    screen = Screen(stdscr)
    stdscr.refresh()
    ircFactory = IRCFactory(screen)
    reactor.addReader(screen)
    reactor.connectTCP('irc.freenode.net', 6667, ircFactory)
    reactor.run()
    screen.close()