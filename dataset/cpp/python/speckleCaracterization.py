from speckleAnalysis import autocorrelation, peakMeasurement
from scipy.signal import convolve2d
import numpy as np


class SpeckleCaracerization:

    def __init__(self, imagePath: str, backgroundImage: str = None, gaussianFilterNormalizationStdDev: float = 75,
                 medianFilterSize: int = 3, imageFromArray: np.ndarray = None):
        self.__fileName = imagePath
        if imageFromArray is not None:
            self.__fileName = "Image from custom array"
        self.__autocorrObj = autocorrelation.Autocorrelation(imagePath, imageFromArray=imageFromArray,
                                                             backgroundImage=backgroundImage)
        self.__image = self.__autocorrObj.image
        self.__autocorrObj.computeAutocorrelation(gaussianFilterNormalizationStdDev, medianFilterSize)
        self.__autocorrelation = self.__autocorrObj.autocorrelation
        self.__verticalSlice, self.__horizontalSlice = self.__autocorrObj.getSlices()
        self.__intensityHistInfo = (None, None, None)
        self.__verticalFWHMFindingMethod = None
        self.__horizontalFWHMFindingMethod = None
        self.__originalParams = (gaussianFilterNormalizationStdDev, medianFilterSize)

    @property
    def speckleImage(self):
        return self.__image

    @property
    def speckleImageAfterFilters(self):
        return self.__autocorrObj.image

    @property
    def fullAutocorrelation(self):
        return self.__autocorrelation

    @property
    def autocorrelationSlices(self):
        return self.__verticalSlice, self.__horizontalSlice

    def crop(self, xStart: int, xEnd: int, yStart: int, yEnd: int, **kwargs):
        newImage = self.__image[int(xStart):int(xEnd), int(yStart):int(yEnd)]
        if kwargs is None:
            return SpeckleCaracerization(
                self.__fileName + f" - Cropped (x = {xStart} to {xEnd}, y = {yStart} to {yEnd})",
                *self.__originalParams, imageFromArray=newImage)
        else:
            return SpeckleCaracerization(
                self.__fileName + f" - Cropped (x = {xStart} to {xEnd}, y = {yStart} to {yEnd})",
                imageFromArray=newImage, **kwargs)

    def centeredCrop(self, width: int, height: int, **kwargs):
        halfWidth = width / 2
        halfHeight = height / 2
        shape = self.__image.shape
        xStart = shape[0] // 2 - np.ceil(halfWidth)
        xEnd = shape[0] // 2 + np.floor(halfWidth)
        yStart = shape[1] // 2 - np.ceil(halfHeight)
        yEnd = shape[1] // 2 + np.ceil(halfHeight)
        return self.crop(xStart, xEnd, yStart, yEnd, **kwargs)

    def computeFWHMOfSpecificAxisWithLinearFit(self, axis: str, maxNbPoints: int = 3, moreInUpperPart: bool = True):
        cleanedAxis = axis.lower().strip()
        if cleanedAxis == "horizontal":
            FWHM = peakMeasurement.FullWidthAtHalfMaximumLinearFit(self.__horizontalSlice, 1, maxNbPoints,
                                                                   moreInUpperPart)
            FWHM_value = FWHM.findFWHM()
            self.__horizontalFWHMFindingMethod = FWHM
        elif cleanedAxis == "vertical":
            FWHM = peakMeasurement.FullWidthAtHalfMaximumLinearFit(self.__verticalSlice, 1, maxNbPoints,
                                                                   moreInUpperPart)
            FWHM_value = FWHM.findFWHM()
            self.__verticalFWHMFindingMethod = FWHM
        else:
            raise ValueError(f"Axis '{axis}' not supported. Try 'horizontal' or 'vertical'.")
        return FWHM_value

    def computeFWHMOfSpecificAxisWithNeighborsAveraging(self, axis: str, averageRange: float = 0.2):
        cleanedAxis = axis.lower().strip()
        if cleanedAxis == "horizontal":
            FWHM = peakMeasurement.FullWidthAtHalfMaximumNeighborsAveraging(self.__horizontalSlice, 1, averageRange)
            FWHM_value = FWHM.findFWHM()
            self.__horizontalFWHMFindingMethod = FWHM
        elif cleanedAxis == "vertical":
            FWHM = peakMeasurement.FullWidthAtHalfMaximumNeighborsAveraging(self.__verticalSlice, 1, averageRange)
            FWHM_value = FWHM.findFWHM()
            self.__verticalFWHMFindingMethod = FWHM
        else:
            raise ValueError(f"Axis '{axis}' not supported. Try 'horizontal' or 'vertical'.")
        return FWHM_value

    def computeFWHMBothAxes(self, method: str = "mean", *args, **kwargs):
        cleanedMethod = method.lower().strip()
        if cleanedMethod == "linear":
            vertical = self.computeFWHMOfSpecificAxisWithLinearFit("vertical", *args, **kwargs)
            horizontal = self.computeFWHMOfSpecificAxisWithLinearFit("horizontal", *args, **kwargs)
        elif cleanedMethod == "mean":
            vertical = self.computeFWHMOfSpecificAxisWithNeighborsAveraging("vertical", *args, **kwargs)
            horizontal = self.computeFWHMOfSpecificAxisWithNeighborsAveraging("horizontal", *args, **kwargs)
        else:
            raise ValueError(f"Method '{method}' not supported. Try 'linear' or 'mean'.")
        return vertical, horizontal

    def intensityHistogram(self, nbBins: int = 256):
        hist, bins = np.histogram(self.__image.ravel(), nbBins, (0, self.__maxPossibleIntensityValue()))
        self.__intensityHistInfo = (hist, bins, nbBins)
        return hist, bins

    def isFullyDevelopedSpecklePattern(self, nbBins: int = 256):
        if self.__intensityHistInfo[-1] != nbBins:
            self.intensityHistogram(nbBins)
        hist, bins, _ = self.__intensityHistInfo
        if np.argmax(hist) == 0:  # If the maximum of the intensity distribution is at index 0, we suppose exp dist.
            return True
        return False

    def meanIntensity(self):
        return np.mean(self.__image).item()

    def stdDevIntensity(self):
        return np.std(self.__image).item()

    def medianIntensity(self):
        return np.median(self.__image).item()

    def maxIntensity(self):
        return np.max(self.__image).item()

    def minIntensity(self):
        return np.min(self.__image).item()

    def contrastModulation(self):
        return (self.maxIntensity() - self.minIntensity()) / (self.maxIntensity() + self.minIntensity())

    def globalContrast(self):
        return self.stdDevIntensity() / self.meanIntensity()

    def localContrast(self, kernelSize: int = 7):
        if kernelSize < 2:
            raise ValueError("The size of the local contrast kernel must be at least 2.")
        kernel = np.ones((kernelSize, kernelSize))
        n = kernel.size
        tempImage = self.__image.astype(float)
        # Put image in float 64 bits, because there can be overflows otherwise
        windowedAverage = convolve2d(kernel, tempImage, "valid") / n
        squaredImageFilter = convolve2d(kernel, tempImage ** 2, "valid")
        # Compute de sample standard deviation
        stdImageWindowed = ((squaredImageFilter - n * windowedAverage ** 2) / (n - 1)) ** 0.5
        return stdImageWindowed / windowedAverage

    def localContrastHistogram(self, nbBins: int = 256, kernelSize: int = 7):
        contrastImage = self.localContrast(kernelSize)
        hist, bins = np.histogram(contrastImage.ravel(), nbBins)
        return hist, bins

    def __maxPossibleIntensityValue(self):
        dtype = self.__image.dtype
        if "float" in str(dtype):
            maxPossible = 1
        elif "int" in str(dtype):
            maxPossible = np.iinfo(dtype).max
        else:
            raise TypeError(f"The type '{dtype}' is not supported for a speckle image.")
        return maxPossible

    def FWHMFindingMethodInfo(self):
        msg = f"Vertical FWHM finding method : {str(self.__verticalFWHMFindingMethod)}\n"
        msg += f"Horizontal FWHM finding method : {str(self.__horizontalFWHMFindingMethod)}"
        return msg


if __name__ == '__main__':
    path = r"C:\Users\goubi\PycharmProjects\HiLoZebrafish\SpeckleSizeCode\MATLAB\\"
    path += r"20190924-200ms_20mW_Ave15_Gray_10X0.4_18.tif"
    path = r"C:\Users\goubi\Desktop\testSpeckle.jpg"
    path = r"C:\Users\goubi\Desktop\Maîtrise\SpeckleData\202009 21-23\20200923-LiquidFITC-Speckles"
    path += r"\20200923-liquidFITC-Speckles-1-8.tif"
    car = SpeckleCaracerization(path)
    carCropped = car.centeredCrop(300, 300, gaussianFilterNormalizationStdDev=0)
    print(carCropped.computeFWHMBothAxes())
