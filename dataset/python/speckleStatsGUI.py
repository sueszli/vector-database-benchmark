from tkinter import filedialog, Tk, ttk, END, StringVar, messagebox, DISABLED, NORMAL, Text, Toplevel
from dcclab.speckleAnalysis import speckleStatsReport, utils
import matplotlib.pyplot as plt
import warnings
import numpy as np

infoFile = "paramsInfo.json"
paramsInfo = utils.jsonToDict(infoFile)
gaussianStdHelp = paramsInfo["gaussianStdHelp"]
medianFilterSizeHelp = paramsInfo["medianFilterSizeHelp"]
localContrastSizeHelp = paramsInfo["localContrastSizeHelp"]
intensityHistBinsHelp = paramsInfo["intensityHistBinsHelp"]
localContrastBinsHelp = paramsInfo["localContrastBinsHelp"]
methodHelp = paramsInfo["methodHelp"]
methodParamsHelp = paramsInfo["methodParamsHelp"]


class SpeckleStatsGUI(Tk):

    def __init__(self, *args, **kwargs):
        super(SpeckleStatsGUI, self).__init__(*args, **kwargs)
        self.protocol('WM_DELETE_WINDOW', self.__close_app)
        # self.resizable(False, False)
        self.title("Speckle Analysis DCCLab")
        header = ttk.Frame(self)
        header.pack()
        self.chooseFileButton = ttk.Button(header, text="Speckle image chooser", command=self.__speckleImageChooser)
        self.chooseFileButton.grid(column=0, row=0, padx=30, pady=30)
        self.filename = None
        self.tabPane = ttk.Notebook(self)
        self.tabPane.pack(expand=1, fill="both")
        self.__parametersTab()
        self.__speckleReport = None
        self.__tabsWidgets = []
        self.__tabsFigures = []
        self.__fullReportPreviewButton = ttk.Button(header, text="Full report preview",
                                                    command=self.__fullReportPreview)
        self.__fullReportPreviewButton["state"] = DISABLED
        self.__fullReportPreviewButton.grid(column=1, row=0, padx=30, pady=30)
        self.__saveFullReportButton = ttk.Button(header, text="Save report to PDF", command=self.__saveFullReport)
        self.__saveFullReportButton["state"] = DISABLED
        self.__saveFullReportButton.grid(column=2, row=0, padx=30, pady=30)
        self.__croppedLims = None

    def __speckleImageChooser(self):
        supportedFiles = [("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.tif;*.tiff")]
        speckleImagePath = filedialog.askopenfilename(title="Please select a speckle image...",
                                                      filetypes=supportedFiles)
        self.__changeImageEvent(speckleImagePath)

    def __changeImageEvent(self, fname: str, newImage: np.ndarray = None):
        if newImage is not None:
            self.__clearTabPaneExceptFirst()
            self.filename = fname
            self.title(f"Speckle Analysis DCCLab ({self.filename})")
        if fname is not None and fname != "":
            self.__clearTabPaneExceptFirst()
            self.filename = fname
            self.title(f"Speckle Analysis DCCLab ({self.filename})")
            self.continueButton["state"] = NORMAL

    def __parametersTab(self):
        methodVar = StringVar(self)
        FWHMFindingMethodParamTextDefault = "Error range for including neighbors (0 to 1) : "

        paramsTab = ttk.Frame(self.tabPane)
        self.tabPane.add(paramsTab, text="Analysis parameters")

        gaussStdLabel = ttk.Label(paramsTab, text="Gaussian normalization filter standard deviation : ")
        gaussStdLabel.grid(column=0, row=0, padx=30, pady=30)
        gaussianStdDev = ttk.Entry(paramsTab)
        gaussianStdDev.insert(END, "75")
        gaussianStdDev.grid(column=1, row=0, padx=30, pady=30)
        utils.ToolTipBind(gaussStdLabel, gaussianStdHelp)

        medianFilterLabel = ttk.Label(paramsTab, text="Median filter size : ")
        medianFilterLabel.grid(column=0, row=1, padx=30, pady=30)
        medianFilterSize = ttk.Entry(paramsTab)
        medianFilterSize.insert(END, "3")
        medianFilterSize.grid(column=1, row=1, padx=30, pady=30)
        utils.ToolTipBind(medianFilterLabel, medianFilterSizeHelp)

        localContrastSizeLabel = ttk.Label(paramsTab, text="Local contrast kernel size : ")
        localContrastSizeLabel.grid(column=0, row=2, padx=30, pady=30)
        localContrastKernelSize = ttk.Entry(paramsTab)
        localContrastKernelSize.insert(END, "7")
        localContrastKernelSize.grid(column=1, row=2, padx=30, pady=30)
        utils.ToolTipBind(localContrastSizeLabel, localContrastSizeHelp)

        intensityHistLabel = ttk.Label(paramsTab, text="Intensity histogram number of bins : ")
        intensityHistLabel.grid(column=2, row=0, padx=30, pady=30)
        nbBinsIntensityHist = ttk.Entry(paramsTab)
        nbBinsIntensityHist.insert(END, "256")
        nbBinsIntensityHist.grid(column=3, row=0, padx=30, pady=30)
        utils.ToolTipBind(intensityHistLabel, intensityHistBinsHelp)

        localContrastBinsLabel = ttk.Label(paramsTab, text="Local contrast histogram number of bins : ")
        localContrastBinsLabel.grid(column=2, row=1, padx=30, pady=30)
        nbBinsLocalContrast = ttk.Entry(paramsTab)
        nbBinsLocalContrast.insert(END, "256")
        nbBinsLocalContrast.grid(column=3, row=1, padx=30, pady=30)
        utils.ToolTipBind(localContrastBinsLabel, localContrastBinsHelp)

        methodLabel = ttk.Label(paramsTab, text="FWHM/diameter finding method : ")
        methodLabel.grid(column=2, row=2, padx=30, pady=30)
        choices = ["Neighbors average", "Linear fit"]
        method = ttk.OptionMenu(paramsTab, methodVar, choices[0], *choices)
        method.grid(column=3, row=2, padx=30, pady=30)
        utils.ToolTipBind(methodLabel, methodHelp)

        FWHMFindingParamLabel = ttk.Label(paramsTab, text=FWHMFindingMethodParamTextDefault)
        FWHMFindingParamLabel.grid(column=2, row=3, padx=30, pady=30)
        FWHMFindingMethodParam = ttk.Entry(paramsTab)
        FWHMFindingMethodParam.insert(END, "0.2")
        FWHMFindingMethodParam.grid(column=3, row=3, padx=30, pady=30)
        utils.ToolTipBind(FWHMFindingParamLabel, methodParamsHelp)

        def onFWHMFindingMethodChange(*args):
            if methodVar.get() == "Neighbors average":
                FWHMFindingParamLabel["text"] = FWHMFindingMethodParamTextDefault
                FWHMFindingMethodParam.delete(0, END)
                FWHMFindingMethodParam.insert(END, "0.2")
            else:
                FWHMFindingParamLabel["text"] = "Maximum number of points for the fit : "
                FWHMFindingMethodParam.delete(0, END)
                FWHMFindingMethodParam.insert(END, "10")

        methodVar.trace("w", onFWHMFindingMethodChange)

        def continueButtonMethod(*args,  ):
            method = methodVar.get()
            if method == "Neighbors average":
                method = "mean"
                methodParamName = "averageRange"
                supposedType = float
            else:
                method = "linear"
                methodParamName = "maxNbPoints"
                supposedType = int
            allParamsKwargs = {"imagePath": self.filename, "gaussianFilterNormalizationStdDev": gaussianStdDev.get(),
                               "medianFilterSize": medianFilterSize.get(),
                               "localContrastKernelSize": localContrastKernelSize.get(),
                               "intensityHistogramBins": nbBinsIntensityHist.get(),
                               "localContrastHistogramBins": nbBinsLocalContrast.get(), "FWHMFindingMethod": method,
                               methodParamName: FWHMFindingMethodParam.get()}
            supposedTypes = [str, float, int, int, int, int, str, supposedType]
            allValid = self.validateEntries(allParamsKwargs, supposedTypes)
            if allValid:
                if self.__speckleReport is not None:
                    self.__clearTabPaneExceptFirst()
                self.__speckleAnalysis(allParamsKwargs)

        self.continueButton = ttk.Button(paramsTab, text="Continue", command=continueButtonMethod, state=DISABLED)
        self.continueButton.grid(column=2, row=4, padx=30, pady=30)

    @classmethod
    def validateEntries(cls, entries: dict, supposedTypes: list):
        allValid = True
        allMsg = []
        for index, key in enumerate(entries):
            value = entries[key]
            supposedType = supposedTypes[index]
            entry, msg = cls.validateType(value, key, supposedType)
            entries[key] = entry
            if msg is not None:
                allMsg.append(msg)
        if len(allMsg) != 0:
            msg = '\n'.join(allMsg)
            messagebox.showerror("Invalid paramters(s)", msg)
            allValid = False
        return allValid

    @classmethod
    def validateType(cls, entry: str, paramName: str, supposedType: type):
        entryRightType = None
        msg = None
        try:
            if entry.strip() == "":
                entry = "0"
            entryRightType = supposedType(entry)
        except:
            msg = f"Parameter '{paramName}' of value {entry} cannot be interpreted as '{supposedType}'."
        return entryRightType, msg

    def __speckleAnalysis(self, kwargs: dict):
        self.withdraw()
        self.__progressBarsRoot = Toplevel()
        self.__progressBarsRoot.title("Please wait")
        self.__progressBarsRoot.geometry("%dx%d%+d%+d" % (249, 81, 250, 125))
        ttk.Label(self.__progressBarsRoot, text="Generating stats...").grid(column=0, row=0, padx=10, pady=10)
        self.__progressBarsRoot.update()
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.__speckleReport = speckleStatsReport.SpeckleStatsReport(**kwargs)
                w = {str(warn): warn.message for warn in w}  # Sometimes two warnings from the "same" source
        except Exception as e:
            messagebox.showerror("Oops!", str(e))
            self.__progressBarsRoot.destroy()
            self.deiconify()
            return
        for warn in w:
            messagebox.showwarning("Watch out!", w[warn])
        ttk.Label(self.__progressBarsRoot, text="Done!").grid(column=1, row=0, padx=10, pady=10)
        ttk.Label(self.__progressBarsRoot, text="Generating visuals...").grid(column=0, row=1, padx=10, pady=10)
        generatingVisuals = ttk.Progressbar(self.__progressBarsRoot, orient="horizontal", length=100,
                                            mode="determinate")
        generatingVisuals["maximum"] = 3
        generatingVisuals.grid(column=1, row=1, padx=10, pady=10)
        self.__progressBarsRoot.update()
        speckleImageStatsTab = self.__speckleImageStatsTab()
        generatingVisuals["value"] = 1
        self.__progressBarsRoot.update()
        speckleAutocorrStatsTab = self.__speckleAutocorrelationStatsTab()
        generatingVisuals["value"] = 2
        self.__progressBarsRoot.update()
        localContrastStatsTab = self.__localContrastStatsTab()
        generatingVisuals["value"] = 3
        self.__progressBarsRoot.update()
        self.__fullReportPreviewButton["state"] = NORMAL
        self.__saveFullReportButton["state"] = NORMAL
        self.tabPane.add(speckleImageStatsTab, text=f"Speckle image stats")
        self.tabPane.add(speckleAutocorrStatsTab, text=f"Speckle autocorrelation stats")
        self.tabPane.add(localContrastStatsTab, text=f"Speckle local contrast stats")
        self.tabPane.select(speckleImageStatsTab)
        self.deiconify()
        self.__progressBarsRoot.destroy()

    def onXLimsChange(self, event):
        print(event.get_xlim())
        self.__croppedLims[0] = event.get_xlim()

    def onYLimsChange(self, event):
        print(event.get_ylim())
        self.__croppedLims[1] = event.get_ylim()

    def __speckleImageStatsTab(self):
        speckleImageStatsTab = ttk.Frame(self.tabPane)
        speckleImageImages = ttk.Frame(speckleImageStatsTab)
        speckleImageDisplay = ttk.Frame(speckleImageImages)
        intensityHistogram = ttk.Frame(speckleImageImages)
        imageFig = plt.figure()
        imageAx = imageFig.add_subplot(111)
        self.__speckleReport._displaySpeckleImagePrep(imageAx, "gray")
        self.__croppedLims = [imageAx.get_xlim(), imageAx.get_ylim()]
        embedImage = utils.MatplotlibFigureEmbedder(speckleImageDisplay, imageFig)
        imageDetachButton = ttk.Button(speckleImageImages, text="Detach/crop", command=self.__speckleImageDetach)
        histDetachButton = ttk.Button(speckleImageImages, text="Detach", command=self.__intensityHistDetach)
        imageHist = plt.figure()
        histAx = imageHist.add_subplot(111)
        self.__speckleReport._intensityHistogramDisplayPrep(histAx)
        embedHist = utils.MatplotlibFigureEmbedder(intensityHistogram, imageHist)
        imageStatsText = self.__speckleReport.speckleImageStats()
        imageStats = Text(speckleImageStatsTab, height=8, width=100)
        imageStats.insert(END, imageStatsText)
        imageStats["state"] = DISABLED
        speckleImageImages.pack()
        speckleImageDisplay.grid(column=1, row=0, padx=5, pady=5)
        intensityHistogram.grid(column=2, row=0, padx=5, pady=5)
        embedImage.embed(False)
        imageDetachButton.grid(column=0, row=0, padx=5, pady=5)
        histDetachButton.grid(column=3, row=0, padx=5, pady=5)
        embedHist.embed(False)
        imageStats.pack()
        self.__tabsWidgets.extend([speckleImageStatsTab, speckleImageImages, speckleImageDisplay, intensityHistogram,
                                   imageDetachButton, histDetachButton, imageStats])
        self.__tabsFigures.extend([imageFig, imageHist])
        return speckleImageStatsTab

    def __speckleAutocorrelationStatsTab(self):
        speckleAutocorrStatsTab = ttk.Frame(self.tabPane)
        specklAutocorrImages = ttk.Frame(speckleAutocorrStatsTab)
        speckleAutocorrDisplay = ttk.Frame(specklAutocorrImages)
        speckleAutocorrSlices = ttk.Frame(specklAutocorrImages)
        autocorrFig = plt.figure()
        autocorrFigAxe = autocorrFig.add_subplot(111)
        self.__speckleReport._displayAutocorrPrep(autocorrFigAxe, None, True, autocorrFig)
        embedImage = utils.MatplotlibFigureEmbedder(speckleAutocorrDisplay, autocorrFig)
        autocorrDetachButton = ttk.Button(specklAutocorrImages, text="Detach", command=self.__fullAutocorrDetach)
        slicesDetachButton = ttk.Button(specklAutocorrImages, text="Detach", command=self.__autocorrSlicesDetach)
        slicesFig = plt.figure()
        ax1 = slicesFig.add_subplot(211)
        ax2 = slicesFig.add_subplot(212)
        self.__speckleReport._displayAutocorrSlicesPrep(slicesFig, ax1, ax2)
        embedAutocorrSlices = utils.MatplotlibFigureEmbedder(speckleAutocorrSlices, slicesFig)
        autocorrStatsText = self.__speckleReport.specklesStats()
        autocorrStats = Text(speckleAutocorrStatsTab, height=8, width=100)
        autocorrStats.insert(END, autocorrStatsText)
        autocorrStats["state"] = DISABLED
        specklAutocorrImages.pack()
        speckleAutocorrDisplay.grid(column=1, row=0, padx=5, pady=5)
        speckleAutocorrSlices.grid(column=2, row=0, padx=5, pady=5)
        embedImage.embed(False)
        autocorrDetachButton.grid(column=0, row=0, padx=5, pady=5)
        slicesDetachButton.grid(column=3, row=0, padx=5, pady=5)
        embedAutocorrSlices.embed(False)
        autocorrStats.pack()
        self.__tabsWidgets.extend(
            [speckleAutocorrStatsTab, specklAutocorrImages, speckleAutocorrDisplay, speckleAutocorrSlices,
             autocorrDetachButton, slicesDetachButton, autocorrStats])
        self.__tabsFigures.extend([autocorrFig, slicesFig])
        return speckleAutocorrStatsTab

    def __localContrastStatsTab(self):
        localContrastStatsTab = ttk.Frame(self.tabPane)
        localContrastImages = ttk.Frame(localContrastStatsTab)
        localContrastImage = ttk.Frame(localContrastImages)
        localContrastHist = ttk.Frame(localContrastImages)
        localContrastFig = plt.figure()
        localContrastAx = localContrastFig.add_subplot(111)
        self.__speckleReport._displayLocalContrastPrep(localContrastAx, None)
        embedImage = utils.MatplotlibFigureEmbedder(localContrastImage, localContrastFig)
        imageDetachButton = ttk.Button(localContrastImages, text="Detach", command=self.__localContrastDetach)
        histDetachButton = ttk.Button(localContrastImages, text="Detach", command=self.__localContrastHistDetach)
        imageHist = plt.figure()
        histAx = imageHist.add_subplot(111)
        self.__speckleReport._localContrastHistogramDisplayPrep(histAx)
        embedHist = utils.MatplotlibFigureEmbedder(localContrastHist, imageHist)
        imageStatsText = self.__speckleReport.localContrastStats()
        imageStats = Text(localContrastStatsTab, height=8, width=100)
        imageStats.insert(END, imageStatsText)
        imageStats["state"] = DISABLED
        localContrastImages.pack()
        localContrastImage.grid(column=1, row=0, padx=5, pady=5)
        localContrastHist.grid(column=2, row=0, padx=5, pady=5)
        embedImage.embed(False)
        imageDetachButton.grid(column=0, row=0, padx=5, pady=5)
        histDetachButton.grid(column=3, row=0, padx=5, pady=5)
        embedHist.embed(False)
        imageStats.pack()
        self.__tabsWidgets.extend([localContrastStatsTab, localContrastImages, localContrastImage, localContrastHist,
                                   imageDetachButton, histDetachButton, imageStats])
        self.__tabsFigures.extend([localContrastFig, imageHist])
        return localContrastStatsTab

    def __fullReportPreview(self):
        msg = "When saving the report to pdf, the layout changes a little to fit into 8.5 inches by 11 inches."
        messagebox.showwarning("Final display", message=msg)
        self.__speckleReport.fullGraphicsReportDisplay()

    def __saveFullReport(self):

        savedFname = filedialog.asksaveasfilename(title="Save report...",
                                                  filetypes=[("Portable Document FIle", "*.pdf")])
        if not savedFname.endswith(".pdf"):
            savedFname += ".pdf"
        self.__speckleReport.fullGrahicsReportCreation(savedFname)
        return savedFname

    def __localContrastHistDetach(self):
        self.__speckleReport.displayLocalContrastHistogram()

    def __localContrastDetach(self):
        self.__speckleReport.displayLocalContrast()

    def __speckleImageDetach(self):
        self.__speckleReport.displaySpeckleImage("gray", self.onYLimsChange, self.onXLimsChange)

    def __intensityHistDetach(self):
        self.__speckleReport.displayIntensityHistogram()

    def __fullAutocorrDetach(self):
        self.__speckleReport.displayFullAutocorrelation()

    def __autocorrSlicesDetach(self):
        self.__speckleReport.displayAutocorrelationSlices()

    def __clearTabPaneExceptFirst(self):
        # Clear everything and destroy every widget present (prevents as many memory leaks as possible)
        self.__fullReportPreviewButton["state"] = DISABLED
        self.__saveFullReportButton["state"] = DISABLED
        for tab in self.tabPane.tabs()[1:]:
            self.tabPane.forget(tab)
        for widget in self.__tabsWidgets:
            widget.destroy()
        self.__tabsWidgets.clear()
        for figure in self.__tabsFigures:
            plt.close(figure)
        plt.close("all")
        self.__tabsFigures.clear()

    def __close_app(self):
        if messagebox.askokcancel("Close", "Are you sure you want to quit? All unsaved progress will be lost."):
            self.quit()


def start():
    app = SpeckleStatsGUI()
    app.mainloop()


if __name__ == '__main__':
    start()
