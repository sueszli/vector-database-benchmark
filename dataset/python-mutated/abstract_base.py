import ctypes
from typing import List, Tuple
from abc import ABC, abstractmethod
QrCodePoint = Tuple[int, int]
QrCodePointList = List[QrCodePoint]

class QrCodeResult:
    """
    A detected QR code.
    """

    def __init__(self, data: str, center: QrCodePoint, points: QrCodePointList):
        if False:
            i = 10
            return i + 15
        self.data: str = data
        self.center: QrCodePoint = center
        self.points: QrCodePointList = points

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return 'data: {} center: {} points: {}'.format(self.data, self.center, self.points)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.data)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.data == other.data

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self == other

class AbstractQrCodeReader(ABC):
    """
    Abstract base class for QR code readers.
    """

    def interval(self) -> float:
        if False:
            return 10
        ' Reimplement to specify a time (in seconds) that the implementation\n        recommends elapse between subsequent calls to read_qr_code.\n        Implementations that have very expensive and/or slow detection code\n        may want to rate-limit read_qr_code calls by overriding this function.\n        e.g.: to make detection happen every 200ms, you would return 0.2 here.\n        Defaults to 0.0'
        return 0.0

    @abstractmethod
    def read_qr_code(self, buffer: ctypes.c_void_p, buffer_size: int, rowlen_bytes: int, width: int, height: int, frame_id: int=-1) -> List[QrCodeResult]:
        if False:
            while True:
                i = 10
        '\n        Reads a QR code from an image buffer in Y800 / GREY format.\n        Returns a list of detected QR codes which includes their data and positions.\n        '