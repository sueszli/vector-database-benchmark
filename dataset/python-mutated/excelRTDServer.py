"""Excel IRTDServer implementation.

This module is a functional example of how to implement the IRTDServer interface
in python, using the pywin32 extensions. Further details, about this interface
and it can be found at:
     http://msdn.microsoft.com/library/default.asp?url=/library/en-us/dnexcl2k2/html/odc_xlrtdfaq.asp
"""
import datetime
import threading
import pythoncom
import win32com.client
from win32com import universal
from win32com.client import gencache
from win32com.server.exception import COMException
EXCEL_TLB_GUID = '{00020813-0000-0000-C000-000000000046}'
EXCEL_TLB_LCID = 0
EXCEL_TLB_MAJOR = 1
EXCEL_TLB_MINOR = 4
gencache.EnsureModule(EXCEL_TLB_GUID, EXCEL_TLB_LCID, EXCEL_TLB_MAJOR, EXCEL_TLB_MINOR)
universal.RegisterInterfaces(EXCEL_TLB_GUID, EXCEL_TLB_LCID, EXCEL_TLB_MAJOR, EXCEL_TLB_MINOR, ['IRtdServer', 'IRTDUpdateEvent'])

class ExcelRTDServer:
    """Base RTDServer class.

    Provides most of the features needed to implement the IRtdServer interface.
    Manages topic adding, removal, and packing up the values for excel.

    Shouldn't be instanciated directly.

    Instead, descendant classes should override the CreateTopic() method.
    Topic objects only need to provide a GetValue() function to play nice here.
    The values given need to be atomic (eg. string, int, float... etc).

    Also note: nothing has been done within this class to ensure that we get
    time to check our topics for updates. I've left that up to the subclass
    since the ways, and needs, of refreshing your topics will vary greatly. For
    example, the sample implementation uses a timer thread to wake itself up.
    Whichever way you choose to do it, your class needs to be able to wake up
    occaisionally, since excel will never call your class without being asked to
    first.

    Excel will communicate with our object in this order:
      1. Excel instanciates our object and calls ServerStart, providing us with
         an IRTDUpdateEvent callback object.
      2. Excel calls ConnectData when it wants to subscribe to a new "topic".
      3. When we have new data to provide, we call the UpdateNotify method of the
         callback object we were given.
      4. Excel calls our RefreshData method, and receives a 2d SafeArray (row-major)
         containing the Topic ids in the 1st dim, and the topic values in the
         2nd dim.
      5. When not needed anymore, Excel will call our DisconnectData to
         unsubscribe from a topic.
      6. When there are no more topics left, Excel will call our ServerTerminate
         method to kill us.

    Throughout, at undetermined periods, Excel will call our Heartbeat
    method to see if we're still alive. It must return a non-zero value, or
    we'll be killed.

    NOTE: By default, excel will at most call RefreshData once every 2 seconds.
          This is a setting that needs to be changed excel-side. To change this,
          you can set the throttle interval like this in the excel VBA object model:
            Application.RTD.ThrottleInterval = 1000 ' milliseconds
    """
    _com_interfaces_ = ['IRtdServer']
    _public_methods_ = ['ConnectData', 'DisconnectData', 'Heartbeat', 'RefreshData', 'ServerStart', 'ServerTerminate']
    _reg_clsctx_ = pythoncom.CLSCTX_INPROC_SERVER
    ALIVE = 1
    NOT_ALIVE = 0

    def __init__(self):
        if False:
            while True:
                i = 10
        'Constructor'
        super().__init__()
        self.IsAlive = self.ALIVE
        self.__callback = None
        self.topics = {}

    def SignalExcel(self):
        if False:
            return 10
        'Use the callback we were given to tell excel new data is available.'
        if self.__callback is None:
            raise COMException(desc='Callback excel provided is Null')
        self.__callback.UpdateNotify()

    def ConnectData(self, TopicID, Strings, GetNewValues):
        if False:
            return 10
        'Creates a new topic out of the Strings excel gives us.'
        try:
            self.topics[TopicID] = self.CreateTopic(Strings)
        except Exception as why:
            raise COMException(desc=str(why))
        GetNewValues = True
        result = self.topics[TopicID]
        if result is None:
            result = '# %s: Waiting for update' % self.__class__.__name__
        else:
            result = result.GetValue()
        self.OnConnectData(TopicID)
        return (result, GetNewValues)

    def DisconnectData(self, TopicID):
        if False:
            while True:
                i = 10
        'Deletes the given topic.'
        self.OnDisconnectData(TopicID)
        if TopicID in self.topics:
            self.topics[TopicID] = None
            del self.topics[TopicID]

    def Heartbeat(self):
        if False:
            print('Hello World!')
        "Called by excel to see if we're still here."
        return self.IsAlive

    def RefreshData(self, TopicCount):
        if False:
            print('Hello World!')
        'Packs up the topic values. Called by excel when it\'s ready for an update.\n\n        Needs to:\n          * Return the current number of topics, via the "ByRef" TopicCount\n          * Return a 2d SafeArray of the topic data.\n            - 1st dim: topic numbers\n            - 2nd dim: topic values\n\n        We could do some caching, instead of repacking everytime...\n        But this works for demonstration purposes.'
        TopicCount = len(self.topics)
        self.OnRefreshData()
        results = [[None] * TopicCount, [None] * TopicCount]
        for (idx, topicdata) in enumerate(self.topics.items()):
            (topicNum, topic) = topicdata
            results[0][idx] = topicNum
            results[1][idx] = topic.GetValue()
        return (tuple(results), TopicCount)

    def ServerStart(self, CallbackObject):
        if False:
            return 10
        'Excel has just created us... We take its callback for later, and set up shop.'
        self.IsAlive = self.ALIVE
        if CallbackObject is None:
            raise COMException(desc='Excel did not provide a callback')
        IRTDUpdateEventKlass = win32com.client.CLSIDToClass.GetClass('{A43788C1-D91B-11D3-8F39-00C04F3651B8}')
        self.__callback = IRTDUpdateEventKlass(CallbackObject)
        self.OnServerStart()
        return self.IsAlive

    def ServerTerminate(self):
        if False:
            print('Hello World!')
        'Called when excel no longer wants us.'
        self.IsAlive = self.NOT_ALIVE
        self.OnServerTerminate()

    def CreateTopic(self, TopicStrings=None):
        if False:
            while True:
                i = 10
        'Topic factory method. Subclass must override.\n\n        Topic objects need to provide:\n          * GetValue() method which returns an atomic value.\n\n        Will raise NotImplemented if not overridden.\n        '
        raise NotImplemented('Subclass must implement')

    def OnConnectData(self, TopicID):
        if False:
            return 10
        "Called when a new topic has been created, at excel's request."
        pass

    def OnDisconnectData(self, TopicID):
        if False:
            print('Hello World!')
        "Called when a topic is about to be deleted, at excel's request."
        pass

    def OnRefreshData(self):
        if False:
            return 10
        'Called when excel has requested all current topic data.'
        pass

    def OnServerStart(self):
        if False:
            print('Hello World!')
        'Called when excel has instanciated us.'
        pass

    def OnServerTerminate(self):
        if False:
            while True:
                i = 10
        'Called when excel is about to destroy us.'
        pass

class RTDTopic:
    """Base RTD Topic.
    Only method required by our RTDServer implementation is GetValue().
    The others are more for convenience."""

    def __init__(self, TopicStrings):
        if False:
            print('Hello World!')
        super().__init__()
        self.TopicStrings = TopicStrings
        self.__currentValue = None
        self.__dirty = False

    def Update(self, sender):
        if False:
            while True:
                i = 10
        'Called by the RTD Server.\n        Gives us a chance to check if our topic data needs to be\n        changed (eg. check a file, quiz a database, etc).'
        raise NotImplemented('subclass must implement')

    def Reset(self):
        if False:
            while True:
                i = 10
        'Call when this topic isn\'t considered "dirty" anymore.'
        self.__dirty = False

    def GetValue(self):
        if False:
            i = 10
            return i + 15
        return self.__currentValue

    def SetValue(self, value):
        if False:
            return 10
        self.__dirty = True
        self.__currentValue = value

    def HasChanged(self):
        if False:
            return 10
        return self.__dirty

class TimeServer(ExcelRTDServer):
    """Example Time RTD server.

    Sends time updates back to excel.

    example of use, in an excel sheet:
      =RTD("Python.RTD.TimeServer","","seconds","5")

    This will cause a timestamp string to fill the cell, and update its value
    every 5 seconds (or as close as possible depending on how busy excel is).

    The empty string parameter denotes the com server is running on the local
    machine. Otherwise, put in the hostname to look on. For more info
    on this, lookup the Excel help for its "RTD" worksheet function.

    Obviously, you'd want to wrap this kind of thing in a friendlier VBA
    function.

    Also, remember that the RTD function accepts a maximum of 28 arguments!
    If you want to pass more, you may need to concatenate arguments into one
    string, and have your topic parse them appropriately.
    """
    _reg_clsid_ = '{EA7F2CF1-11A2-45E4-B2D5-68E240DB8CB1}'
    _reg_progid_ = 'Python.RTD.TimeServer'
    _reg_desc_ = 'Python class implementing Excel IRTDServer -- feeds time'
    INTERVAL = 0.5

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.ticker = threading.Timer(self.INTERVAL, self.Update)

    def OnServerStart(self):
        if False:
            while True:
                i = 10
        self.ticker.start()

    def OnServerTerminate(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.ticker.finished.isSet():
            self.ticker.cancel()

    def Update(self):
        if False:
            return 10
        self.ticker = threading.Timer(self.INTERVAL, self.Update)
        try:
            if len(self.topics):
                refresh = False
                for topic in self.topics.values():
                    topic.Update(self)
                    if topic.HasChanged():
                        refresh = True
                    topic.Reset()
                if refresh:
                    self.SignalExcel()
        finally:
            self.ticker.start()

    def CreateTopic(self, TopicStrings=None):
        if False:
            for i in range(10):
                print('nop')
        'Topic factory. Builds a TimeTopic object out of the given TopicStrings.'
        return TimeTopic(TopicStrings)

class TimeTopic(RTDTopic):
    """Example topic for example RTD server.

    Will accept some simple commands to alter how long to delay value updates.

    Commands:
      * seconds, delay_in_seconds
      * minutes, delay_in_minutes
      * hours, delay_in_hours
    """

    def __init__(self, TopicStrings):
        if False:
            return 10
        super().__init__(TopicStrings)
        try:
            (self.cmd, self.delay) = self.TopicStrings
        except Exception as E:
            raise ValueError('Invalid topic strings: %s' % str(TopicStrings))
        self.delay = float(self.delay)
        self.checkpoint = self.timestamp()
        self.SetValue(str(self.checkpoint))

    def timestamp(self):
        if False:
            while True:
                i = 10
        return datetime.datetime.now()

    def Update(self, sender):
        if False:
            print('Hello World!')
        now = self.timestamp()
        delta = now - self.checkpoint
        refresh = False
        if self.cmd == 'seconds':
            if delta.seconds >= self.delay:
                refresh = True
        elif self.cmd == 'minutes':
            if delta.minutes >= self.delay:
                refresh = True
        elif self.cmd == 'hours':
            if delta.hours >= self.delay:
                refresh = True
        else:
            self.SetValue('#Unknown command: ' + self.cmd)
        if refresh:
            self.SetValue(str(now))
            self.checkpoint = now
if __name__ == '__main__':
    import win32com.server.register
    win32com.server.register.UseCommandLine(TimeServer)