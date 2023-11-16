"""
Positioning interfaces.

@since: 14.0
"""
from zope.interface import Attribute, Interface

class IPositioningReceiver(Interface):
    """
    An interface for positioning providers.
    """

    def positionReceived(latitude, longitude):
        if False:
            print('Hello World!')
        '\n        Method called when a position is received.\n\n        @param latitude: The latitude of the received position.\n        @type latitude: L{twisted.positioning.base.Coordinate}\n        @param longitude: The longitude of the received position.\n        @type longitude: L{twisted.positioning.base.Coordinate}\n        '

    def positionErrorReceived(positionError):
        if False:
            i = 10
            return i + 15
        '\n        Method called when position error is received.\n\n        @param positionError: The position error.\n        @type positionError: L{twisted.positioning.base.PositionError}\n        '

    def timeReceived(time):
        if False:
            i = 10
            return i + 15
        '\n        Method called when time and date information arrives.\n\n        @param time: The date and time (expressed in UTC unless otherwise\n            specified).\n        @type time: L{datetime.datetime}\n        '

    def headingReceived(heading):
        if False:
            for i in range(10):
                print('nop')
        '\n        Method called when a true heading is received.\n\n        @param heading: The heading.\n        @type heading: L{twisted.positioning.base.Heading}\n        '

    def altitudeReceived(altitude):
        if False:
            while True:
                i = 10
        '\n        Method called when an altitude is received.\n\n        @param altitude: The altitude.\n        @type altitude: L{twisted.positioning.base.Altitude}\n        '

    def speedReceived(speed):
        if False:
            return 10
        '\n        Method called when the speed is received.\n\n        @param speed: The speed of a mobile object.\n        @type speed: L{twisted.positioning.base.Speed}\n        '

    def climbReceived(climb):
        if False:
            while True:
                i = 10
        '\n        Method called when the climb is received.\n\n        @param climb: The climb of the mobile object.\n        @type climb: L{twisted.positioning.base.Climb}\n        '

    def beaconInformationReceived(beaconInformation):
        if False:
            for i in range(10):
                print('nop')
        '\n        Method called when positioning beacon information is received.\n\n        @param beaconInformation: The beacon information.\n        @type beaconInformation: L{twisted.positioning.base.BeaconInformation}\n        '

class IPositioningBeacon(Interface):
    """
    A positioning beacon.
    """
    identifier = Attribute('\n        A unique identifier for this beacon. The type is dependent on the\n        implementation, but must be immutable.\n        ')

class INMEAReceiver(Interface):
    """
    An object that can receive NMEA data.
    """

    def sentenceReceived(sentence):
        if False:
            return 10
        '\n        Method called when a sentence is received.\n\n        @param sentence: The received NMEA sentence.\n        @type L{twisted.positioning.nmea.NMEASentence}\n        '
__all__ = ['IPositioningReceiver', 'IPositioningBeacon', 'INMEAReceiver']