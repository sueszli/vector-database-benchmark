import speckleAnalysis.speckleCaracterization as caracterization
from speckleAnalysis.utils import twoListsIntersection, sortedAlphanumeric
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class FilesFinder:

    def __init__(self, directoryPath: str):
        self.__dirPath = directoryPath
        self.__files = os.listdir(self.__dirPath)

    def __getCompletePath(self, filepath: str):
        return os.path.join(self.__dirPath, filepath)

    def returnAllFiles(self, relativePathOnly: bool = True):
        allFiles = self.__files
        if relativePathOnly:
            return allFiles
        return [self.__getCompletePath(filepath) for filepath in allFiles]

    def returnSpecificExtensions(self, extensions: tuple, relativePathOnly: bool = True):
        allFiles = self.__files
        if relativePathOnly:
            return [filepath for filepath in allFiles if filepath.endswith(extensions)]
        return [self.__getCompletePath(filepath) for filepath in allFiles if filepath.endswith(extensions)]

    def returnSpecificFilesContainingSpecificKeywords(self, keywords: tuple, relativePathOnly: bool = True):
        allFiles = self.__files
        filesToReturn = []
        for filepath in allFiles:
            for keyword in keywords:
                if keyword in filepath:
                    if relativePathOnly:
                        filesToReturn.append(filepath)
                    else:
                        filesToReturn.append(self.__getCompletePath(filepath))
                    break
        return filesToReturn


class SpeckleFolderCaracterizations:

    def __init__(self, directoryPath: str, backgroundImage: str = None, specificKeywords: tuple = None,
                 specificExtensions: tuple = None, cropAroundCenter: tuple = None, **kwargs):
        self.__allFilesObj = FilesFinder(directoryPath)
        if specificExtensions is not None and specificKeywords is not None:
            allFilesExtensions = self.__allFilesObj.returnSpecificExtensions(specificExtensions, False)
            allFilesKeyword = self.__allFilesObj.returnSpecificFilesContainingSpecificKeywords(specificKeywords, False)
            self.__allFiles = twoListsIntersection(allFilesExtensions, allFilesKeyword)
        elif specificKeywords is not None:
            self.__allFiles = self.__allFilesObj.returnSpecificFilesContainingSpecificKeywords(specificKeywords, False)
        elif specificExtensions is not None:
            self.__allFiles = self.__allFilesObj.returnSpecificExtensions(specificExtensions, False)
        else:
            self.__allFiles = self.__allFilesObj.returnAllFiles(False)
        self.__allFiles = sortedAlphanumeric(self.__allFiles)
        self.__dataTile = directoryPath
        self.__caracKwargs = kwargs
        self.__allCaracObj = None
        self.__cropSize = cropAroundCenter
        self.__bgImage = backgroundImage

    def __createAllCaracterizationObjects(self):
        kwargs = self.__caracKwargs
        if self.__cropSize is None:
            self.__allCaracObj = [caracterization.SpeckleCaracerization(file, self.__bgImage, **kwargs) for file in
                                  self.__allFiles]
        else:
            self.__allCaracObj = [
                caracterization.SpeckleCaracerization(file, self.__bgImage, **kwargs).centeredCrop(self.__cropSize[0],
                                                                                                   self.__cropSize[1])
                for file in self.__allFiles]

    def allDiameters(self, method: str = "mean", *args, **kwargs):
        if self.__allCaracObj is None:
            self.__createAllCaracterizationObjects()
        diams = [sum(carObj.computeFWHMBothAxes(method, *args, **kwargs)) / 2 for carObj in self.__allCaracObj]
        return diams

    def allGlobalContrasts(self):
        if self.__allCaracObj is None:
            self.__createAllCaracterizationObjects()
        return [carObj.globalContrast() for carObj in self.__allCaracObj]

    def allContrastModulations(self):
        if self.__allCaracObj is None:
            self.__createAllCaracterizationObjects()
        return [carObj.contrastModulation() for carObj in self.__allCaracObj]

    def allFullyDeveloped(self):
        if self.__allCaracObj is None:
            self.__createAllCaracterizationObjects()
        return [carObj.isFullyDevelopedSpecklePattern() for carObj in self.__allCaracObj]

    def allDataToCSV(self, filename: str, fileSeparator: str = ",", **diametersComputationKwargs):
        index, allDiams, allGContrasts, allVisibilities, allFullyDeveloped = self.allData(**diametersComputationKwargs)

        columns = [f"Diameter [px]", "Global contrasts", "Contrast modulation / visibility",
                   "is fully developed (1:yes, 0:no)? (256 bins)"]
        data = np.vstack([allDiams, allGContrasts, allVisibilities, allFullyDeveloped]).T
        dframe = pd.DataFrame(data, columns=columns, index=index)
        dframe.to_csv(filename, fileSeparator, index=True)

    def allData(self, **diametersComputationKwargs):
        diams = []
        globalContrasts = []
        visibilities = []
        fullyDeveloped = []
        caracObjs = []
        shortFnames = []
        nbFiles = len(self.__allFiles)
        i = 0
        for file in self.__allFiles:
            sName = os.path.split(file)[-1]
            shortFnames.append(sName)
            try:
                obj = caracterization.SpeckleCaracerization(file, self.__bgImage, **self.__caracKwargs)
                if self.__cropSize is not None:
                    obj = obj.centeredCrop(self.__cropSize[0], self.__cropSize[1], **self.__caracKwargs)
                im = obj.speckleImageAfterFilters
                # plt.imshow(im, cmap="gray")
                # plt.show()
                caracObjs.append(obj)
                diams.append(sum(obj.computeFWHMBothAxes(**diametersComputationKwargs)) / 2)
                globalContrasts.append(obj.globalContrast())
                visibilities.append(obj.contrastModulation())
                fullyDeveloped.append(obj.isFullyDevelopedSpecklePattern())
            except Exception as e:
                diams.append(np.nan)
                globalContrasts.append(np.nan)
                visibilities.append(np.nan)
                fullyDeveloped.append(-1)
                print(f"Problem with {sName}")
                print(e)
            i += 1
            print(f"Number of patterns treated : {i}/{nbFiles}")
        self.__allCaracObj = caracObjs
        return shortFnames, diams, globalContrasts, visibilities, fullyDeveloped


if __name__ == '__main__':
    fullPath = r"C:\Users\goubi\Desktop\Maîtrise\SpeckleData\20201016-FITCSpeckles-0p5Objective\20201016-Speckles-2"
    dirpath = r"C:\Users\goubi\Desktop\Maîtrise\SpeckleData\202009 21-23\20200923-LiquidFITC-Speckles"
    speckleInfo = SpeckleFolderCaracterizations(fullPath, cropAroundCenter=(300, 300),
                                                gaussianFilterNormalizationStdDev=75, medianFilterSize=0)
    speckleInfo.allDataToCSV("20201016-FITCSpeckles-0p5Objective_20201016-Speckles-2_75std.csv", averageRange=0.3)
