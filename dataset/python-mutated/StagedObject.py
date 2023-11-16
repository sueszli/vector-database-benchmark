class StagedObject:
    """
    Use this class as a mixin to provide an interface for onStage/offStage objects.

    The idea here is that a DistributedObject could be present and active due to
    simple visibility, but we want to hide or otherwise disable it for some reason.
    """
    UNKNOWN = -1
    OFF = 0
    ON = 1

    def __init__(self, initState=UNKNOWN):
        if False:
            for i in range(10):
                print('nop')
        '\n        Only sets the initial state of this object.  This will not\n        call any "handle" functions.\n        '
        self.__state = initState

    def goOnStage(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        '\n        If a stage switch is needed, the correct "handle" function\n        will be called.  Otherwise, nothing happens.\n        '
        if not self.isOnStage():
            self.handleOnStage(*args, **kw)

    def handleOnStage(self):
        if False:
            while True:
                i = 10
        "\n        Override this function to provide your on/off stage funcitionality.\n\n        Don't forget to call down to this one, though.\n        "
        self.__state = StagedObject.ON

    def goOffStage(self, *args, **kw):
        if False:
            return 10
        '\n        If a stage switch is needed, the correct "handle" function\n        will be called.  Otherwise, nothing happens.\n        '
        if not self.isOffStage():
            self.handleOffStage(*args, **kw)

    def handleOffStage(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Override this function to provide your on/off stage funcitionality.\n\n        Don't forget to call down to this one, though.\n        "
        self.__state = StagedObject.OFF

    def isOnStage(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__state == StagedObject.ON

    def isOffStage(self):
        if False:
            while True:
                i = 10
        return self.__state == StagedObject.OFF