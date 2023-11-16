from impacket.structure import Structure

class MSDS_MANAGEDPASSWORD_BLOB(Structure):
    structure = (('Version', '<H'), ('Reserved', '<H'), ('Length', '<L'), ('CurrentPasswordOffset', '<H'), ('PreviousPasswordOffset', '<H'), ('QueryPasswordIntervalOffset', '<H'), ('UnchangedPasswordIntervalOffset', '<H'), ('CurrentPassword', ':'), ('PreviousPassword', ':'), ('QueryPasswordInterval', ':'), ('UnchangedPasswordInterval', ':'))

    def __init__(self, data=None):
        if False:
            return 10
        Structure.__init__(self, data=data)

    def fromString(self, data):
        if False:
            i = 10
            return i + 15
        Structure.fromString(self, data)
        if self['PreviousPasswordOffset'] == 0:
            endData = self['QueryPasswordIntervalOffset']
        else:
            endData = self['PreviousPasswordOffset']
        self['CurrentPassword'] = self.rawData[self['CurrentPasswordOffset']:][:endData - self['CurrentPasswordOffset']]
        if self['PreviousPasswordOffset'] != 0:
            self['PreviousPassword'] = self.rawData[self['PreviousPasswordOffset']:][:self['QueryPasswordIntervalOffset'] - self['PreviousPasswordOffset']]
        self['QueryPasswordInterval'] = self.rawData[self['QueryPasswordIntervalOffset']:][:self['UnchangedPasswordIntervalOffset'] - self['QueryPasswordIntervalOffset']]
        self['UnchangedPasswordInterval'] = self.rawData[self['UnchangedPasswordIntervalOffset']:]