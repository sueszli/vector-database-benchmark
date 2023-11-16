from dcclab.speckleAnalysis import speckleCaracterization
import matplotlib.pyplot as plt
import numpy as np


class SpeckleStatsReport:

    def __init__(self, imagePath: str, backgroundImage: str = None, gaussianFilterNormalizationStdDev: float = 75,
                 medianFilterSize: int = 3, localContrastKernelSize: int = 7, intensityHistogramBins: int = 256,
                 localContrastHistogramBins: int = 256, FWHMFindingMethod: str = "mean", *FWHMFindingMethodArgs,
                 **FWHMFindingMethodKwargs):
        self.__imagePath = imagePath
        self.__speckleCaracterizationObj = speckleCaracterization.SpeckleCaracerization(imagePath, backgroundImage,
                                                                                        gaussianFilterNormalizationStdDev,
                                                                                        medianFilterSize)
        self.verticalFWHM, self.horizontalFWHM = self.__speckleCaracterizationObj.computeFWHMBothAxes(
            FWHMFindingMethod, *FWHMFindingMethodArgs, **FWHMFindingMethodKwargs)
        self.__autocorrelation = self.__speckleCaracterizationObj.fullAutocorrelation
        self.__verticalSlice, self.__horizontalSlice = self.__speckleCaracterizationObj.autocorrelationSlices
        self.__image = self.__speckleCaracterizationObj.speckleImage
        self.__localContrast = self.__speckleCaracterizationObj.localContrast(localContrastKernelSize)
        self.__localContrastHist, self.__localContrastBins = self.__speckleCaracterizationObj.localContrastHistogram(
            localContrastHistogramBins, localContrastKernelSize)
        self.__localContrastKernelSize = localContrastKernelSize
        self.__intensityHist, self.__intensityBins = self.__speckleCaracterizationObj.intensityHistogram(
            intensityHistogramBins)
        self.isFullyDevelopped = self.__speckleCaracterizationObj.isFullyDevelopedSpecklePattern(intensityHistogramBins)
        self.meanIntensity = self.__speckleCaracterizationObj.meanIntensity()
        self.stdDevIntensity = self.__speckleCaracterizationObj.stdDevIntensity()
        self.medianIntensity = self.__speckleCaracterizationObj.medianIntensity()
        self.maxIntensity = self.__speckleCaracterizationObj.maxIntensity()
        self.minIntensity = self.__speckleCaracterizationObj.minIntensity()
        self.contrastModulation = self.__speckleCaracterizationObj.contrastModulation()
        self.globalContrast = self.__speckleCaracterizationObj.globalContrast()
        self.__fullReport = None

    @property
    def speckleImage(self):
        return self.__image

    @property
    def fullAutocorrelation(self):
        return self.__autocorrelation

    @property
    def autocorrelationSlices(self):
        return self.__verticalSlice, self.__horizontalSlice

    @property
    def localContrast(self):
        return self.__localContrast

    @classmethod
    def __imageDisplayPrep(cls, axis, image: np.ndarray, title: str, colorMap: str):
        image = axis.imshow(image, colorMap)
        axis.set_title(title)
        return axis, image

    @classmethod
    def __plotDisplayPrep(cls, axis, data: np.ndarray, title: str, xlabel: str, ylabel: str = None):
        axis.plot(data)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        if ylabel is not None:
            axis.set_ylabel(ylabel)
        return axis

    @classmethod
    def __histogramDisplayPrep(cls, axis, data: np.ndarray, title: str, xlabel: str, ylabel: str, bins):
        if data.ndim != 1:
            data = data.ravel()
        axis.hist(data, bins)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        return axis

    def _intensityHistogramDisplayPrep(self, axis):
        title = f"Intensity histogram\n({len(self.__intensityBins) - 1} bins, ranging from {self.__intensityBins[0]} to"
        title += f" {self.__intensityBins[-1]})"
        return self.__histogramDisplayPrep(axis, self.__image, title, "Intensity value [-]", "Count [-]",
                                           self.__intensityBins)

    def _localContrastHistogramDisplayPrep(self, axis):
        title = f"Local contrast histogram\n({len(self.__localContrastBins) - 1} bins, ranging"
        title += f" from {np.round(self.__localContrastBins[0], 2)} to {np.round(self.__localContrastBins[-1], 2)})"
        self.__histogramDisplayPrep(axis, self.__localContrast, title, "Local contrast value [-]", "Count [-]",
                                    self.__localContrastBins)

    def _displaySpeckleImagePrep(self, axis, colorMap: str):
        return self.__imageDisplayPrep(axis, self.__image, "Speckle image", colorMap)

    def _displayAutocorrPrep(self, axis, colorMap: str, colorBar: bool, fig=None):
        ax, image = self.__imageDisplayPrep(axis, self.__autocorrelation, "Full normalized autocorrelation", colorMap)
        if colorBar:
            if fig is None:
                raise ValueError("You must provide the figure to display the colorbar.")
            fig.colorbar(image)
        return ax

    def _displayLocalContrastPrep(self, axis, colorMap: str):
        return self.__imageDisplayPrep(axis, self.__localContrast,
                                       f"Local contrast (kernel size of {self.__localContrastKernelSize})", colorMap)

    def _displayAutocorrSlicesPrep(self, fig, axisForHorizontal, axisForVertical):
        ax1 = axisForHorizontal
        ax2 = axisForVertical
        fig.suptitle("Autocorrelation slices")

        self.__plotDisplayPrep(ax1, self.__horizontalSlice, "Central horizontal slice",
                               "Horizontal position $x$ [pixel]")

        self.__plotDisplayPrep(ax2, self.__verticalSlice, "Central vertical slice", "Vertical position $y$ [pixel]")

        ylabel = "Normalized autocorrelation coefficient [-]"
        fig.text(0.04, 0.5, ylabel, ha='center', va='center', rotation='vertical')
        fig.subplots_adjust(hspace=0.6)

    def displaySpeckleImage(self, colorMap: str = None, connectYZoom: callable = None, connectXZoom: callable = None):
        fig = plt.figure()
        ax = fig.add_subplot()
        self._displaySpeckleImagePrep(ax, colorMap)
        if connectXZoom is not None:
            ax.callbacks.connect('xlim_changed', connectXZoom)
        if connectYZoom is not None:
            ax.callbacks.connect('ylim_changed', connectYZoom)
        fig.show()

    def displayFullAutocorrelation(self, colorMap: str = None, showColorBar: bool = True):
        fig = plt.figure()
        ax = fig.add_subplot()
        self._displayAutocorrPrep(ax, colorMap, showColorBar, fig)
        fig.show()

    def displayAutocorrelationSlices(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        self._displayAutocorrSlicesPrep(fig, ax1, ax2)
        fig.show()

    def displayIntensityHistogram(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        self._intensityHistogramDisplayPrep(ax)
        fig.show()

    def displayLocalContrastHistogram(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        self._localContrastHistogramDisplayPrep(ax)
        fig.show()

    def displayLocalContrast(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        self._displayLocalContrastPrep(ax, None)
        fig.show()

    def speckleImageStats(self):
        intenityStats = f"Mean intensity : {self.meanIntensity}\nIntensity std deviation : {self.stdDevIntensity}\n"
        intenityStats += f"Maximum intensity : {self.maxIntensity}\nMinimum intensity : {self.minIntensity}\n"
        intenityStats += f"Global contrast : {self.globalContrast}\nContrast modulation : {self.contrastModulation}\n"
        fullyDeveloped = "This is not a fully developed speckle pattern\n"
        fullyDeveloped += "(based on the intensity histogram, its maximum is not at 0, not assuming exponential " \
                          "distribution)\n"
        if self.isFullyDevelopped:
            fullyDeveloped = "This is a fully developed speckle pattern\n"
            fullyDeveloped += "(based on the intensity histogram, its maximum is at 0, assuming exponential " \
                              "distribution)\n"
        intenityStats += fullyDeveloped
        return intenityStats

    def specklesStats(self):
        speckleStats = f"Vertical diam. : {self.verticalFWHM} pixels\nHorizontal diam. : {self.horizontalFWHM} pixels\n"
        speckleStats += self.__speckleCaracterizationObj.FWHMFindingMethodInfo() + "\n"
        return speckleStats

    def localContrastStats(self):
        localContrastStats = f"Local contrast mean : {np.mean(self.__localContrast)}\n"
        localContrastStats += f"Local contrast std deviation : {np.std(self.__localContrast)}\n"
        localContrastStats += f"Local contrast median : {np.median(self.__localContrast)}\n"
        localContrastStats += f"Local contrast min : {np.min(self.__localContrast)}\n"
        localContrastStats += f"Local contrast max : {np.max(self.__localContrast)}"
        return localContrastStats

    def textReport(self):
        header = "========== Statistical properties of the speckle image ==========\n"
        midSection = "========== Statistical properties of the speckles ==========\n"
        basicStats = self.speckleImageStats()
        speckleStats = self.specklesStats()
        localContrastStats = self.localContrastStats()
        allText = header + basicStats + midSection + speckleStats + localContrastStats
        return allText

    def __str__(self):
        return self.textReport()

    def fullGrahicsReportCreation(self, saveName: str = None):
        if self.__fullReport is None:
            fig = plt.figure()
            fig.set_size_inches((8.5, 11), forward=False)  # For saving purpose
            ax1 = fig.add_subplot(321)
            ax2 = fig.add_subplot(322)
            ax3 = fig.add_subplot(323)
            ax4 = fig.add_subplot(324)
            ax5 = fig.add_subplot(325)
            ax6 = fig.add_subplot(326)
            axes = np.array([ax5, ax6])
            fig.subplots_adjust(hspace=0.4, wspace=0.5)
            gs = axes[0].get_gridspec()
            for ax in axes:
                ax.remove()
            axBig = fig.add_subplot(gs[:])
            fig.suptitle(f"Speckles statistical report of\n{self.__imagePath}", wrap=True)

            self._displaySpeckleImagePrep(ax1, None)

            self._displayLocalContrastPrep(ax2, None)

            self._intensityHistogramDisplayPrep(ax3)

            self._localContrastHistogramDisplayPrep(ax4)

            text = self.textReport()
            axBig.text(0.5, -0.1, text, ha="center", fontsize=8)
            axBig.axis("off")

            self.__fullReport = fig

        if saveName is not None:
            self.__fullReport.savefig(fname=saveName, dpi=1000)
        return self.__fullReport

    def fullGraphicsReportDisplay(self, saveName: str = None, useInGUI: bool = True):
        fig = self.fullGrahicsReportCreation(saveName)
        if useInGUI:
            fig.show()
        else:
            plt.show()

    def fullMethodInfo(self):
        raise NotImplementedError("To do")


if __name__ == '__main__':
    path = r"..\speckleAnalysis\circularWithPhasesSimulations\4pixelsCircularWithPhasesSimulations.tiff"
    # path = r"C:\Users\goubi\PycharmProjects\HiLoZebrafish\SpeckleSizeCode\MATLAB\\"
    # path += r"20190924-200ms_20mW_Ave15_Gray_10X0.4_20.tif"
    # path = r"C:\Users\goubi\Desktop\testSpeckle.jpg"
    ssr = SpeckleStatsReport(path, averageRange=20 / 100)
    ssr.fullGraphicsReportDisplay(None, False)
