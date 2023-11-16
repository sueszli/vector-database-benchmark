import sys
import threading
import time
from gnuradio import gr
import pmt

class offloadThread(threading.Thread):

    def __init__(self, callback, overlayList, listDelay, repeat):
        if False:
            for i in range(10):
                print('nop')
        threading.Thread.__init__(self)
        self.callback = callback
        self.overlayList = overlayList
        self.listDelay = listDelay
        self.threadRunning = False
        self.stopThread = False
        self.repeat = repeat

    def run(self):
        if False:
            print('Hello World!')
        self.stopThread = False
        self.threadRunning = True
        time.sleep(0.5)
        if type(self.overlayList) == list and self.listDelay > 0.0:
            while self.repeat and (not self.stopThread):
                for curItem in self.overlayList:
                    self.callback(curItem)
                    if self.stopThread:
                        break
                    time.sleep(self.listDelay)
                    if self.stopThread:
                        break
        else:
            self.callback(self.overlayList)
        self.threadRunning = False

class GrGraphicOverlay(gr.sync_block):
    """
    This block is an example of how to feed an overlay to a graphic item.
    The graphic item overlay is expecting a dictionary with the following
    keys: 'filename','x','y', and optionally a 'scalefactor'.  A list of
    dictionaries can also be supplied to support multiple items.

    Any file can be added to the graphic item as an overlay and the
    particular item indexed by its filename can be updated by passing
    in new x/y coordinates.  To remove an overlay, use coordinates -1,-1
    for the x,y coordinates.

    This sample block sends either a dictionary or list of dictionaries
    to the graphicitem block.  To test updating a single overlay item,
    you can use a list with the same file but different coordinates and
    use the update delay > 0.0 to animate it.
    """

    def __init__(self, overlayList, listDelay, repeat):
        if False:
            for i in range(10):
                print('nop')
        gr.sync_block.__init__(self, name='GrGraphicsOverlay', in_sig=None, out_sig=None)
        self.overlayList = overlayList
        self.listDelay = listDelay
        if type(self.overlayList) is not dict and type(self.overlayList) is not list:
            gr.log.error("The specified input is not valid.  Please specify either a dictionary item with the following keys: 'filename','x','y'[,'scalefactor'] or a list of dictionary items.")
            sys.exit(1)
        self.message_port_register_out(pmt.intern('overlay'))
        self.thread = offloadThread(self.overlayCallback, self.overlayList, listDelay, repeat)
        self.thread.start()

    def overlayCallback(self, msgData):
        if False:
            i = 10
            return i + 15
        meta = pmt.to_pmt(msgData)
        pdu = pmt.cons(meta, pmt.PMT_NIL)
        self.message_port_pub(pmt.intern('overlay'), pdu)

    def stop(self):
        if False:
            while True:
                i = 10
        self.thread.stopThread = True
        while self.thread.threadRunning:
            time.sleep(0.1)
        return True