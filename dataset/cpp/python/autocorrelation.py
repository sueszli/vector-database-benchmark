import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import gaussian_filter, median_filter
import cv2


class FileReader:
    # FIXME: This is only until pydcclab is ok

    @staticmethod
    def readFile(path: str, bgImage: str = None):
        """
        Method used to read an image file. Optionally, this method can be used to remove background. Background removal
        is done by subtracting the background image from the current image we want to read. If a value comes out
        negative, it is clipped to 0.
        :param path: The path of the image we want to analyze.
        :param bgImage: (optional) The path of the background image (set to None by default).
        :return: The image as a NumPy array, with dimensions (width, height).
        """
        if path.endswith(".tif") or path.endswith(".tiff"):
            pixels = tifffile.imread(path)
        else:
            pixels = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if bgImage is not None:
            if bgImage.endswith(".tif") or bgImage.endswith(".tiff"):
                bg = tifffile.imread(path)
            else:
                bg = cv2.imread(bgImage, cv2.IMREAD_GRAYSCALE)
            pixels = pixels - bg
        pixels[pixels < 0] = 0
        return pixels.T  # we want shape[0] as the width and shape[1] as the height


class Autocorrelation:
    # FIXME: Use pydcclab when ok

    def __init__(self, imagePath: str, imageFromArray: np.ndarray = None, backgroundImage:str=None):
        if imageFromArray is not None:
            self.__image = imageFromArray.copy()
        else:
            self.__image = FileReader.readFile(imagePath, backgroundImage)
        self.__original = self.image
        self.__autocorrelation = None
        self.__slicesObj = None

    @property
    def image(self):
        return self.__image.copy().T

    @property
    def original(self):
        return self.__original.copy().T

    @property
    def autocorrelation(self):
        if self.__autocorrelation is None:
            return None
        return self.__autocorrelation.copy().T

    def getSlices(self, indices: tuple = None):
        if self.__slicesObj is None:
            raise ValueError("Please compute the autocorrelation to access its slices.")
        if indices is None:
            return self.__slicesObj.middleSlices()
        return self.__slicesObj.slicesAt(indices)

    def computeAutocorrelation(self, gaussianFilterStdDev: float = 75, medianFilterSize: int = 3):
        self._gaussianNormalization(gaussianFilterStdDev)  # First, gaussian filter normalization
        self._medianFilter(medianFilterSize)  # Then, median filter to remove noise
        self._autocorrelationWithFourierTransform()  # Compute the autocorrelation

    def _gaussianNormalization(self, filterStdDev: float = 75):
        if filterStdDev == 0:
            return
        if filterStdDev < 0:
            raise ValueError("The gaussian filter's standard deviation must be positive and non zero.")
        filteredImage = gaussian_filter(self.__image, filterStdDev)
        self.__image = self.__image / filteredImage - np.mean(self.__image)

    def _autocorrelationWithFourierTransform(self):
        fft = np.fft.fft2(self.__image)
        ifft = np.fft.ifftshift(np.fft.ifft2(np.abs(fft) ** 2)).real
        ifft /= np.size(ifft)
        self.__autocorrelation = (ifft - np.mean(self.__image) ** 2) / np.var(self.__image)
        self.__slicesObj = AutocorrelationSlices(self.__autocorrelation)

    def _medianFilter(self, filterSize: int = 3):
        if filterSize == 0:
            return
        if filterSize < 2:
            raise ValueError("The size of the median filter must be at least 2.")
        self.__image = median_filter(self.__image, filterSize)

    def showImage(self):
        plt.imshow(self.__image)
        plt.show()

    def showAutocorrelation(self, showColorbar: bool = True):
        if self.__autocorrelation is None:
            raise ValueError("No autocorrelation computed.")
        plt.imshow(self.__autocorrelation)
        if showColorbar:
            plt.colorbar()
        plt.show()

    def showAutocorrelationSlices(self, indices: tuple = None, showHorizontal: bool = True, showVertical: bool = True):
        vSlice, hSlice = self.getSlices(indices)
        if showHorizontal and showVertical:
            fig, (ax1, ax2) = plt.subplots(2, sharey="col")
            fig.suptitle("Autocorrelation slices")

            ax1.plot(hSlice)
            ax1.set_title(f"Horizontal slice (at index {indices[0]})")
            ax1.set_xlabel("Horizontal position $x$ [pixel]")

            ax2.plot(vSlice)
            ax2.set_title(f"Vertical slice (at index {indices[1]})")
            ax2.set_xlabel("Vertical position $y$ [pixel]")

            ylabel = "Normalized autocorrelation coefficient [-]"
            fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
            plt.subplots_adjust(hspace=0.32)
            plt.show()
        elif showVertical:
            plt.plot(vSlice)
            plt.title(f"Vertical slice (at index {indices[1]})")
            plt.xlabel("Vertical position $y$ [pixel]")
            plt.ylabel("Normalized autocorrelation coefficient [-]")
            plt.show()
        elif showHorizontal:
            plt.plot(hSlice)
            plt.title(f"Horizontal slice (at index {indices[0]})")
            plt.xlabel("Horizontal position $x$ [pixel]")
            plt.ylabel("Normalized autocorrelation coefficient [-]")
            plt.show()


class AutocorrelationSlices:

    def __init__(self, autocorrelation: np.ndarray):
        if not isinstance(autocorrelation, np.ndarray):
            raise TypeError("The autocorrelation parameter must be a numpy array.")
        if not autocorrelation.ndim == 2:
            raise ValueError("The autocorrelation must be in 2D.")
        self.__autocorrelation = autocorrelation

    def slicesAt(self, indices: tuple):
        if len(indices) != 2:
            raise ValueError("There must be 2 indices of slicing, one horizontal and one vertical.")
        # Assumes indices[0] is the width and indices[1] is the height
        xSlice, ySlice = indices[0], indices[1]
        verticalSlice = self.__autocorrelation[:, ySlice]
        horizontalSlice = self.__autocorrelation[xSlice, :]
        return verticalSlice, horizontalSlice

    def middleSlices(self):
        # Assumes the shape of the autocorrelation is (width, height) or (nb columns, nb lines)
        middleX, middleY = self.__autocorrelation.shape[0] // 2, self.__autocorrelation.shape[1] // 2
        return self.slicesAt((middleX, middleY))
