from matplotlib.backends.backend_tkagg import FigureCanvasAgg
import matplotlib.backends.tkagg as tkagg
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import tkinter as tk
'\nA graph of time to ping Google.com\nDemonstrates Matploylib used in an animated way.\n\nNote this file contains a copy of ping.py. It is contained in the first part of this file\n\n'
'\n    A pure python ping implementation using raw sockets.\n\n    (This is Python 3 port of https://github.com/jedie/python-ping)\n    (Tested and working with python 2.7, should work with 2.6+)\n\n    Note that ICMP messages can only be sent from processes running as root\n    (in Windows, you must run this script as \'Administrator\').\n\n    Derived from ping.c distributed in Linux\'s netkit. That code is\n    copyright (c) 1989 by The Regents of the University of California.\n    That code is in turn derived from code written by Mike Muuss of the\n    US Army Ballistic Research Laboratory in December, 1983 and\n    placed in the public domain. They have my thanks.\n\n    Bugs are naturally mine. I\'d be glad to hear about them. There are\n    certainly word - size dependencies here.\n\n    Copyright (c) Matthew Dixon Cowles, <http://www.visi.com/~mdc/>.\n    Distributable under the terms of the GNU General Public License\n    version 2. Provided with no warranties of any sort.\n\n    Original Version from Matthew Dixon Cowles:\n      -> ftp://ftp.visi.com/users/mdc/ping.py\n\n    Rewrite by Jens Diemer:\n      -> http://www.python-forum.de/post-69122.html#69122\n\n    Rewrite by George Notaras:\n      -> http://www.g-loaded.eu/2009/10/30/python-ping/\n\n    Enhancements by Martin Falatic:\n      -> http://www.falatic.com/index.php/39/pinging-with-python\n\n    Enhancements and fixes by Georgi Kolev:\n      -> http://github.com/jedie/python-ping/\n\n    Bug fix by Andrejs Rozitis:\n      -> http://github.com/rozitis/python-ping/\n\n    Revision history\n    ~~~~~~~~~~~~~~~~\n    May 1, 2014\n    -----------\n    Little modifications by Mohammad Emami <emamirazavi@gmail.com>\n    - Added Python 3 support. For now this project will just support\n      python 3.x\n    - Tested with python 3.3\n    - version was upped to 0.6\n\n    March 19, 2013\n    --------------\n    * Fixing bug to prevent divide by 0 during run-time.\n\n    January 26, 2012\n    ----------------\n    * Fixing BUG #4 - competability with python 2.x [tested with 2.7]\n      - Packet data building is different for 2.x and 3.x.\n        \'cose of the string/bytes difference.\n    * Fixing BUG #10 - the multiple resolv issue.\n      - When pinging domain names insted of hosts (for exmaple google.com)\n        you can get different IP every time you try to resolv it, we should\n        resolv the host only once and stick to that IP.\n    * Fixing BUGs #3 #10 - Doing hostname resolv only once.\n    * Fixing BUG #14 - Removing all \'global\' stuff.\n        - You should not use globul! Its bad for you...and its not thread safe!\n    * Fix - forcing the use of different times on linux/windows for\n            more accurate mesurments. (time.time - linux/ time.clock - windows)\n    * Adding quiet_ping function - This way we\'ll be able to use this script\n        as external lib.\n    * Changing default timeout to 3s. (1second is not enought)\n    * Switching data syze to packet size. It\'s easyer for the user to ignore the\n        fact that the packet headr is 8b and the datasize 64 will make packet with\n        size 72.\n\n    October 12, 2011\n    --------------\n    Merged updates from the main project\n      -> https://github.com/jedie/python-ping\n\n    September 12, 2011\n    --------------\n    Bugfixes + cleanup by Jens Diemer\n    Tested with Ubuntu + Windows 7\n\n    September 6, 2011\n    --------------\n    Cleanup by Martin Falatic. Restored lost comments and docs. Improved\n    functionality: constant time between pings, internal times consistently\n    use milliseconds. Clarified annotations (e.g., in the checksum routine).\n    Using unsigned data in IP & ICMP header pack/unpack unless otherwise\n    necessary. Signal handling. Ping-style output formatting and stats.\n\n    August 3, 2011\n    --------------\n    Ported to py3k by Zach Ware. Mostly done by 2to3; also minor changes to\n    deal with bytes vs. string changes (no more ord() in checksum() because\n    >source_string< is actually bytes, added .encode() to data in\n    send_one_ping()).  That\'s about it.\n\n    March 11, 2010\n    --------------\n    changes by Samuel Stauffer:\n    - replaced time.clock with default_timer which is set to\n      time.clock on windows and time.time on other systems.\n\n    November 8, 2009\n    ----------------\n    Improved compatibility with GNU/Linux systems.\n\n    Fixes by:\n     * George Notaras -- http://www.g-loaded.eu\n    Reported by:\n     * Chris Hallman -- http://cdhallman.blogspot.com\n\n    Changes in this release:\n     - Re-use time.time() instead of time.clock(). The 2007 implementation\n       worked only under Microsoft Windows. Failed on GNU/Linux.\n       time.clock() behaves differently under the two OSes[1].\n\n    [1] http://docs.python.org/library/time.html#time.clock\n\n    May 30, 2007\n    ------------\n    little rewrite by Jens Diemer:\n     -  change socket asterisk import to a normal import\n     -  replace time.time() with time.clock()\n     -  delete "return None" (or change to "return" only)\n     -  in checksum() rename "str" to "source_string"\n\n    December 4, 2000\n    ----------------\n    Changed the struct.pack() calls to pack the checksum and ID as\n    unsigned. My thanks to Jerome Poincheval for the fix.\n\n    November 22, 1997\n    -----------------\n    Initial hack. Doesn\'t do much, but rather than try to guess\n    what features I (or others) will want in the future, I\'ve only\n    put in what I need now.\n\n    December 16, 1997\n    -----------------\n    For some reason, the checksum bytes are in the wrong order when\n    this is run under Solaris 2.X for SPARC but it works right under\n    Linux x86. Since I don\'t know just what\'s wrong, I\'ll swap the\n    bytes always and then do an htons().\n\n    ===========================================================================\n    IP header info from RFC791\n      -> http://tools.ietf.org/html/rfc791)\n\n    0                   1                   2                   3\n    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1\n    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n    |Version|  IHL  |Type of Service|          Total Length         |\n    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n    |         Identification        |Flags|      Fragment Offset    |\n    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n    |  Time to Live |    Protocol   |         Header Checksum       |\n    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n    |                       Source Address                          |\n    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n    |                    Destination Address                        |\n    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n    |                    Options                    |    Padding    |\n    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n\n    ===========================================================================\n    ICMP Echo / Echo Reply Message header info from RFC792\n      -> http://tools.ietf.org/html/rfc792\n\n        0                   1                   2                   3\n        0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1\n        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n        |     Type      |     Code      |          Checksum             |\n        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n        |           Identifier          |        Sequence Number        |\n        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n        |     Data ...\n        +-+-+-+-+-\n\n    ===========================================================================\n    ICMP parameter info:\n      -> http://www.iana.org/assignments/icmp-parameters/icmp-parameters.xml\n\n    ===========================================================================\n    An example of ping\'s typical output:\n\n    PING heise.de (193.99.144.80): 56 data bytes\n    64 bytes from 193.99.144.80: icmp_seq=0 ttl=240 time=127 ms\n    64 bytes from 193.99.144.80: icmp_seq=1 ttl=240 time=127 ms\n    64 bytes from 193.99.144.80: icmp_seq=2 ttl=240 time=126 ms\n    64 bytes from 193.99.144.80: icmp_seq=3 ttl=240 time=126 ms\n    64 bytes from 193.99.144.80: icmp_seq=4 ttl=240 time=127 ms\n\n    ----heise.de PING Statistics----\n    5 packets transmitted, 5 packets received, 0.0% packet loss\n    round-trip (ms)  min/avg/max/med = 126/127/127/127\n\n    ===========================================================================\n'
import argparse
import os, sys, socket, struct, select, time, signal
__description__ = 'A pure python ICMP ping implementation using raw sockets.'
if sys.platform == 'win32':
    default_timer = time.clock
else:
    default_timer = time.time
NUM_PACKETS = 3
PACKET_SIZE = 64
WAIT_TIMEOUT = 3.0
ICMP_ECHOREPLY = 0
ICMP_ECHO = 8
ICMP_MAX_RECV = 2048
MAX_SLEEP = 1000

class MyStats:
    thisIP = '0.0.0.0'
    pktsSent = 0
    pktsRcvd = 0
    minTime = 999999999
    maxTime = 0
    totTime = 0
    avrgTime = 0
    fracLoss = 1.0
myStats = MyStats

def checksum(source_string):
    if False:
        while True:
            i = 10
    '\n    A port of the functionality of in_cksum() from ping.c\n    Ideally this would act on the string as a series of 16-bit ints (host\n    packed), but this works.\n    Network data is big-endian, hosts are typically little-endian\n    '
    countTo = int(len(source_string) / 2) * 2
    sum = 0
    count = 0
    loByte = 0
    hiByte = 0
    while count < countTo:
        if sys.byteorder == 'little':
            loByte = source_string[count]
            hiByte = source_string[count + 1]
        else:
            loByte = source_string[count + 1]
            hiByte = source_string[count]
        try:
            sum = sum + (hiByte * 256 + loByte)
        except:
            sum = sum + (ord(hiByte) * 256 + ord(loByte))
        count += 2
    if countTo < len(source_string):
        loByte = source_string[len(source_string) - 1]
        try:
            sum += loByte
        except:
            sum += ord(loByte)
    sum &= 4294967295
    sum = (sum >> 16) + (sum & 65535)
    sum += sum >> 16
    answer = ~sum & 65535
    answer = socket.htons(answer)
    return answer

def do_one(myStats, destIP, hostname, timeout, mySeqNumber, packet_size, quiet=False):
    if False:
        print('Hello World!')
    '\n    Returns either the delay (in ms) or None on timeout.\n    '
    delay = None
    try:
        mySocket = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.getprotobyname('icmp'))
    except socket.error as e:
        print("failed. (socket error: '%s')" % e.args[1])
        raise
    my_ID = os.getpid() & 65535
    sentTime = send_one_ping(mySocket, destIP, my_ID, mySeqNumber, packet_size)
    if sentTime == None:
        mySocket.close()
        return delay
    myStats.pktsSent += 1
    (recvTime, dataSize, iphSrcIP, icmpSeqNumber, iphTTL) = receive_one_ping(mySocket, my_ID, timeout)
    mySocket.close()
    if recvTime:
        delay = (recvTime - sentTime) * 1000
        if not quiet:
            print('%d bytes from %s: icmp_seq=%d ttl=%d time=%d ms' % (dataSize, socket.inet_ntoa(struct.pack('!I', iphSrcIP)), icmpSeqNumber, iphTTL, delay))
        myStats.pktsRcvd += 1
        myStats.totTime += delay
        if myStats.minTime > delay:
            myStats.minTime = delay
        if myStats.maxTime < delay:
            myStats.maxTime = delay
    else:
        delay = None
        print('Request timed out.')
    return delay

def send_one_ping(mySocket, destIP, myID, mySeqNumber, packet_size):
    if False:
        i = 10
        return i + 15
    '\n    Send one ping to the given >destIP<.\n    '
    myChecksum = 0
    header = struct.pack('!BBHHH', ICMP_ECHO, 0, myChecksum, myID, mySeqNumber)
    padBytes = []
    startVal = 66
    if sys.version[:1] == '2':
        bytes = struct.calcsize('d')
        data = (packet_size - 8 - bytes) * 'Q'
        data = struct.pack('d', default_timer()) + data
    else:
        for i in range(startVal, startVal + (packet_size - 8)):
            padBytes += [i & 255]
        data = bytearray(padBytes)
    myChecksum = checksum(header + data)
    header = struct.pack('!BBHHH', ICMP_ECHO, 0, myChecksum, myID, mySeqNumber)
    packet = header + data
    sendTime = default_timer()
    try:
        mySocket.sendto(packet, (destIP, 1))
    except socket.error as e:
        print('General failure (%s)' % e.args[1])
        return
    return sendTime

def receive_one_ping(mySocket, myID, timeout):
    if False:
        while True:
            i = 10
    '\n    Receive the ping from the socket. Timeout = in ms\n    '
    timeLeft = timeout / 1000
    while True:
        startedSelect = default_timer()
        whatReady = select.select([mySocket], [], [], timeLeft)
        howLongInSelect = default_timer() - startedSelect
        if whatReady[0] == []:
            return (None, 0, 0, 0, 0)
        timeReceived = default_timer()
        (recPacket, addr) = mySocket.recvfrom(ICMP_MAX_RECV)
        ipHeader = recPacket[:20]
        (iphVersion, iphTypeOfSvc, iphLength, iphID, iphFlags, iphTTL, iphProtocol, iphChecksum, iphSrcIP, iphDestIP) = struct.unpack('!BBHHHBBHII', ipHeader)
        icmpHeader = recPacket[20:28]
        (icmpType, icmpCode, icmpChecksum, icmpPacketID, icmpSeqNumber) = struct.unpack('!BBHHH', icmpHeader)
        if icmpPacketID == myID:
            dataSize = len(recPacket) - 28
            return (timeReceived, dataSize + 8, iphSrcIP, icmpSeqNumber, iphTTL)
        timeLeft = timeLeft - howLongInSelect
        if timeLeft <= 0:
            return (None, 0, 0, 0, 0)

def dump_stats(myStats):
    if False:
        i = 10
        return i + 15
    '\n    Show stats when pings are done\n    '
    print('\n----%s PYTHON PING Statistics----' % myStats.thisIP)
    if myStats.pktsSent > 0:
        myStats.fracLoss = (myStats.pktsSent - myStats.pktsRcvd) / myStats.pktsSent
    print('%d packets transmitted, %d packets received, %0.1f%% packet loss' % (myStats.pktsSent, myStats.pktsRcvd, 100.0 * myStats.fracLoss))
    if myStats.pktsRcvd > 0:
        print('round-trip (ms)  min/avg/max = %d/%0.1f/%d' % (myStats.minTime, myStats.totTime / myStats.pktsRcvd, myStats.maxTime))
    print('')
    return

def signal_handler(signum, frame):
    if False:
        for i in range(10):
            print('nop')
    '\n    Handle exit via signals\n    '
    dump_stats()
    print('\n(Terminated with signal %d)\n' % signum)
    sys.exit(0)

def verbose_ping(hostname, timeout=WAIT_TIMEOUT, count=NUM_PACKETS, packet_size=PACKET_SIZE, path_finder=False):
    if False:
        while True:
            i = 10
    '\n    Send >count< ping to >destIP< with the given >timeout< and display\n    the result.\n    '
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)
    myStats = MyStats()
    mySeqNumber = 0
    try:
        destIP = socket.gethostbyname(hostname)
        print('\nPYTHON PING %s (%s): %d data bytes' % (hostname, destIP, packet_size))
    except socket.gaierror as e:
        print('\nPYTHON PING: Unknown host: %s (%s)' % (hostname, e.args[1]))
        print()
        return
    myStats.thisIP = destIP
    for i in range(count):
        delay = do_one(myStats, destIP, hostname, timeout, mySeqNumber, packet_size)
        if delay == None:
            delay = 0
        mySeqNumber += 1
        if MAX_SLEEP > delay:
            time.sleep((MAX_SLEEP - delay) / 1000)
    dump_stats(myStats)

def quiet_ping(hostname, timeout=WAIT_TIMEOUT, count=NUM_PACKETS, packet_size=PACKET_SIZE, path_finder=False):
    if False:
        return 10
    '\n    Same as verbose_ping, but the results are returned as tuple\n    '
    myStats = MyStats()
    mySeqNumber = 0
    try:
        destIP = socket.gethostbyname(hostname)
    except socket.gaierror as e:
        return (0, 0, 0, 0)
    myStats.thisIP = destIP
    if path_finder:
        fakeStats = MyStats()
        do_one(fakeStats, destIP, hostname, timeout, mySeqNumber, packet_size, quiet=True)
        time.sleep(0.5)
    for i in range(count):
        delay = do_one(myStats, destIP, hostname, timeout, mySeqNumber, packet_size, quiet=True)
        if delay == None:
            delay = 0
        mySeqNumber += 1
        if MAX_SLEEP > delay:
            time.sleep((MAX_SLEEP - delay) / 1000)
    if myStats.pktsSent > 0:
        myStats.fracLoss = (myStats.pktsSent - myStats.pktsRcvd) / myStats.pktsSent
    if myStats.pktsRcvd > 0:
        myStats.avrgTime = myStats.totTime / myStats.pktsRcvd
    return (myStats.maxTime, myStats.minTime, myStats.avrgTime, myStats.fracLoss)
SIZE = (320, 240)

class MyGlobals:
    axis_pings = None
    ping_x_array = []
    ping_y_array = []
g_my_globals = MyGlobals()

def run_a_ping_and_graph():
    if False:
        while True:
            i = 10
    global g_my_globals
    response = quiet_ping('google.com', timeout=1000)
    if response[0] == 0:
        ping_time = 1000
    else:
        ping_time = response[0]
    g_my_globals.ping_x_array.append(len(g_my_globals.ping_x_array))
    g_my_globals.ping_y_array.append(ping_time)
    if len(g_my_globals.ping_x_array) > 100:
        x_array = g_my_globals.ping_x_array[-100:]
        y_array = g_my_globals.ping_y_array[-100:]
    else:
        x_array = g_my_globals.ping_x_array
        y_array = g_my_globals.ping_y_array
    g_my_globals.axis_ping.clear()
    set_chart_labels()
    g_my_globals.axis_ping.plot(x_array, y_array)

def set_chart_labels():
    if False:
        while True:
            i = 10
    global g_my_globals
    g_my_globals.axis_ping.set_xlabel('Time', fontsize=8)
    g_my_globals.axis_ping.set_ylabel('Ping (ms)', fontsize=8)
    g_my_globals.axis_ping.set_title('Current Ping Duration', fontsize=8)

def draw(fig, canvas):
    if False:
        i = 10
        return i + 15
    (figure_x, figure_y, figure_w, figure_h) = fig.bbox.bounds
    (figure_w, figure_h) = (int(figure_w), int(figure_h))
    photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)
    canvas.create_image(SIZE[0] / 2, SIZE[1] / 2, image=photo)
    figure_canvas_agg = FigureCanvasAgg(fig)
    figure_canvas_agg.draw()
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)
    return photo

def main():
    if False:
        for i in range(10):
            print('nop')
    global g_my_globals
    layout = [[sg.Canvas(size=SIZE, background_color='white', key='canvas'), sg.Button('Exit', pad=(0, (210, 0)))]]
    window = sg.Window('Ping Graph', layout, background_color='white', grab_anywhere=True, finalize=True)
    canvas_elem = window['canvas']
    canvas = canvas_elem.TKCanvas
    fig = plt.figure(figsize=(3.1, 2.25), tight_layout={'pad': 0})
    g_my_globals.axis_ping = fig.add_subplot(1, 1, 1)
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    set_chart_labels()
    plt.tight_layout()
    while True:
        (event, values) = window.read(timeout=0)
        if event in ('Exit', None):
            break
        run_a_ping_and_graph()
        photo = draw(fig, canvas)
    window.close()
if __name__ == '__main__':
    main()