import numpy as np
import math
import warnings


class HalfWidthAtHalfMaximumOneDimension:

    def __init__(self, data: np.ndarray, maximum: float = None, positiveSlopeSideOfPeak: bool = None):
        # If no maximum is provided, taking the maximal value of data. Else, taking the provided one.
        # if positiveSlopeSideOfPeak is not None: assumes there is only half of the central peak, either the increasing
        # side (arg is True) or the decreasing one (arg is False).
        # Else, the algorithm takes the left left side of the peak.
        if maximum is None:
            maximum = np.max(data)
        self.maximum = maximum
        if positiveSlopeSideOfPeak is None:
            data = data[:np.argmax(data)]
            positiveSlopeSideOfPeak = True
        if not isinstance(data, np.ndarray):
            raise TypeError("The data must be within a numpy array.")
        if data.ndim != 1:
            raise ValueError("The data must be in one dimension.")
        self._data = data
        self._sideOfPeak = "left" if positiveSlopeSideOfPeak else "right"
        self.HWHM = None
        self.report = None

    def findHWHM(self):
        raise NotImplementedError("This method must be implemented by subclasses.")

    def __str__(self):
        return "General HwHM/FWHM finding method"


class HalfWidthAtHalfMaximumNeighborsAveraging(HalfWidthAtHalfMaximumOneDimension):

    def __init__(self, data: np.ndarray, maximum: float = None, positiveSlopeSideOfPeak: bool = None,
                 errorRange: float = 20 / 100):
        if not (0 <= errorRange < 1):
            raise ValueError("The range of neighbors must lie in the half open interval (0, 1].")
        if errorRange > 0.5:
            warnings.warn("A large range can lead to inaccurate results for the HWHM (and diameter) measurement.")
        self.__range = errorRange
        self.__dataUsed = None
        super(HalfWidthAtHalfMaximumNeighborsAveraging, self).__init__(data, maximum, positiveSlopeSideOfPeak)

    def findHWHM(self):
        if self.HWHM is not None:
            msg = "The half width at half maximum is already computed. You can access it with the attribute 'HWHM'."
            warnings.warn(msg, UserWarning)
            return self.HWHM
        range = self.__range
        halfMax = self.maximum / 2
        inferiorBound = halfMax - halfMax * range
        superiorBound = halfMax + halfMax * range
        pointsForHWHM = np.where((self._data >= inferiorBound) & (self._data <= superiorBound))[0]
        nbPoints = len(pointsForHWHM)
        if nbPoints == 0:
            raise ValueError("The range is too small. Not enough values were found to compute the HWHM.")
        mean = np.mean(pointsForHWHM)
        left = 0
        right = mean
        if self._sideOfPeak == "left":
            left = mean
            right = len(self._data)
        HWHM = right - left
        self.HWHM = HWHM
        self.__dataUsed = pointsForHWHM
        return HWHM

    def __str__(self):
        msg = f"Error/neighbors average method (±{self.__range * 100}%).\n"
        msg += "For more info, see the method's 'fullMethodInfo'."
        return msg


class HalfWidthAtHalfMaximumLinearFit(HalfWidthAtHalfMaximumOneDimension):

    def __init__(self, data: np.ndarray, maximum: float = None, positiveSlopeSideOfPeak: bool = None,
                 maxNumberOfPoints: int = 10, moreInUpperPart: bool = True):
        self.__maxNbPts = maxNumberOfPoints
        self.__dataUsed = None
        self.__moreInUpperPart = moreInUpperPart
        self.__fitInfo = None
        super(HalfWidthAtHalfMaximumLinearFit, self).__init__(data, maximum, positiveSlopeSideOfPeak)

    def findHWHM(self):
        if self.HWHM is not None:
            msg = "The half width at half maximum is already computed. You can access it with the attribute 'HWHM'."
            warnings.warn(msg, UserWarning)
            return self.HWHM
        maxNbPoints = self.__maxNbPts
        moreInUpperPart = self.__moreInUpperPart
        if maxNbPoints < 2:
            raise ValueError("There should be at least 2 points for the linear fit.")
        halfMax = self.maximum / 2
        halfK = maxNbPoints / 2
        upperKs = math.ceil(halfK)
        lowerKs = math.floor(halfK)
        if not moreInUpperPart:
            temp = upperKs
            upperKs = lowerKs
            lowerKs = temp
        lows, highs, lIndices, hIndices = splitInTwoWithMiddleValue(halfMax, self._data, True)
        if self._sideOfPeak == "left":
            lows = lows[-lowerKs:]
            highs = highs[:upperKs]
            lIndices = lIndices[-lowerKs:]
            hIndices = hIndices[:upperKs]
        else:
            lows = lows[:lowerKs]
            highs = highs[-upperKs:]
            lIndices = lIndices[:lowerKs]
            hIndices = hIndices[-upperKs:]
        xData = np.append(lIndices, hIndices)
        yData = np.append(lows, highs)
        (slope, zero), covMat = np.polyfit(xData, yData, 1, full=False,
                                           cov=True)
        left = findXWithY(halfMax, slope, zero)
        right = len(self._data)
        if self._sideOfPeak == "right":
            right = findXWithY(halfMax, slope, zero)
            left = 0
        HWHM = right - left
        self.HWHM = HWHM
        self.__dataUsed = (lows, highs, lIndices, hIndices)
        self.__fitInfo = (slope, zero, covMat)
        return HWHM

    def __str__(self):
        msg = f"Linear fit method (max of {self.__maxNbPts} points).\n"
        msg += "For more info, see the method's 'fullMethodInfo'."
        return msg


class FullWidthAtHalfMaximumNeighborsAveraging(HalfWidthAtHalfMaximumNeighborsAveraging):

    def __init__(self, data: np.ndarray, maximum: float = None, errorRange: float = 20 / 100):
        self.FWHM = None
        super(FullWidthAtHalfMaximumNeighborsAveraging, self).__init__(data, maximum, errorRange=errorRange)

    def findFWHM(self):
        self.FWHM = self.findHWHM() * 2
        return self.FWHM

    def __str__(self):
        msgBase = super(FullWidthAtHalfMaximumNeighborsAveraging, self).__str__()
        msgBase += "** Info only on the HWHM finding **"
        return msgBase


class FullWidthAtHalfMaximumLinearFit(HalfWidthAtHalfMaximumLinearFit):
    def __init__(self, data: np.ndarray, maximum: float = None, maximumNumberOfPoints: int = 10,
                 moreInUpperPart: bool = True):
        self.FWHM = None
        super(FullWidthAtHalfMaximumLinearFit, self).__init__(data, maximum, None, maximumNumberOfPoints,
                                                              moreInUpperPart)

    def findFWHM(self):
        self.FWHM = self.findHWHM() * 2
        return self.FWHM

    def __str__(self):
        msgBase = super(FullWidthAtHalfMaximumLinearFit, self).__str__()
        msgBase += "** Info only on the HWHM finding **"
        return msgBase


def splitInTwoWithMiddleValue(middleValue: float, array: np.ndarray, returnIndices: bool = False):
    # Assumes the values are only increasing or decreasing, not both.
    # Excludes the middle value
    upper = np.ravel(np.where(array > middleValue))  # Doesn't change anything since 1D data
    lower = np.ravel(np.where(array < middleValue))  # Doesn't change anything since 1D data
    lowerValues = array[lower]
    upperValues = array[upper]
    if not returnIndices:
        return lowerValues, upperValues
    return lowerValues, upperValues, lower, upper


def findXWithY(y, slope, zero):
    x = (y - zero) / slope
    return x
