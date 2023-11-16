from direct.directnotify import DirectNotifyGlobal
from direct.distributed.ClockDelta import globalClockDelta
from direct.distributed import DistributedObjectAI

class TimeManagerAI(DistributedObjectAI.DistributedObjectAI):
    notify = DirectNotifyGlobal.directNotify.newCategory('TimeManagerAI')

    def __init__(self, air):
        if False:
            for i in range(10):
                print('nop')
        DistributedObjectAI.DistributedObjectAI.__init__(self, air)

    def requestServerTime(self, context):
        if False:
            i = 10
            return i + 15
        'requestServerTime(self, int8 context)\n\n        This message is sent from the client to the AI to initiate a\n        synchronization phase.  The AI should immediately report back\n        with its current time.  The client will then measure the round\n        trip.\n        '
        timestamp = globalClockDelta.getRealNetworkTime(bits=32)
        requesterId = self.air.getAvatarIdFromSender()
        print('requestServerTime from %s' % requesterId)
        self.sendUpdateToAvatarId(requesterId, 'serverTime', [context, timestamp])