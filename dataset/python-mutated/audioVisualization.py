from __future__ import print_function
import shutil, struct, simplejson
from scipy.spatial import distance
from pylab import *
import ntpath
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioTrainTest as aT
import sklearn
import sklearn.discriminant_analysis
import sys
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def generateColorMap():
    if False:
        for i in range(10):
            print('nop')
    '\n    This function generates a 256 jet colormap of HTML-like\n    hex string colors (e.g. FF88AA)\n    '
    Map = cm.jet(np.arange(256))
    stringColors = []
    for i in range(Map.shape[0]):
        rgb = (int(255 * Map[i][0]), int(255 * Map[i][1]), int(255 * Map[i][2]))
        if sys.version_info > (3, 0):
            stringColors.append(struct.pack('BBB', *rgb).hex())
        else:
            stringColors.append(struct.pack('BBB', *rgb).encode('hex'))
    return stringColors

def levenshtein(str1, s2):
    if False:
        i = 10
        return i + 15
    '\n    Distance between two strings\n    '
    N1 = len(str1)
    N2 = len(s2)
    stringRange = [range(N1 + 1)] * (N2 + 1)
    for i in range(N2 + 1):
        stringRange[i] = range(i, i + N1 + 1)
    for i in range(0, N2):
        for j in range(0, N1):
            if str1[j] == s2[i]:
                stringRange[i + 1][j + 1] = min(stringRange[i + 1][j] + 1, stringRange[i][j + 1] + 1, stringRange[i][j])
            else:
                stringRange[i + 1][j + 1] = min(stringRange[i + 1][j] + 1, stringRange[i][j + 1] + 1, stringRange[i][j] + 1)
    return stringRange[N2][N1]

def text_list_to_colors(names):
    if False:
        print('Hello World!')
    '\n    Generates a list of colors based on a list of names (strings). \n    Similar strings correspond to similar colors.\n    '
    Dnames = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            Dnames[i, j] = 1 - 2.0 * levenshtein(names[i], names[j]) / float(len(names[i] + names[j]))
    pca = sklearn.decomposition.PCA(n_components=1)
    pca.fit(Dnames)
    textToColor = pca.transform(Dnames)
    textToColor = 255 * (textToColor - textToColor.min()) / (textToColor.max() - textToColor.min())
    textmaps = generateColorMap()
    colors = [textmaps[int(c)] for c in textToColor]
    return colors

def text_list_to_colors_simple(names):
    if False:
        i = 10
        return i + 15
    '\n    Generates a list of colors based on a list of names (strings). \n    Similar strings correspond to similar colors. \n    '
    uNames = list(set(names))
    uNames.sort()
    textToColor = [uNames.index(n) for n in names]
    textToColor = np.array(textToColor)
    textToColor = 255 * (textToColor - textToColor.min()) / (textToColor.max() - textToColor.min())
    textmaps = generateColorMap()
    colors = [textmaps[int(c)] for c in textToColor]
    return colors

def visualizeFeaturesFolder(folder, dimReductionMethod, priorKnowledge='none'):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function generates a  content visualization for the recordings\n     of the provided path.\n    ARGUMENTS:\n        - folder:        path of the folder that contains the WAV files \n                         to be processed\n        - dimReductionMethod:    method used to reduce the dimension of the \n                                 initial feature space before computing \n                                 the similarity.\n        - priorKnowledge:    if this is set equal to "artist"\n    '
    if dimReductionMethod == 'pca':
        (all_mt_feat, wav_files, _) = aF.directory_feature_extraction(folder, 30.0, 30.0, 0.05, 0.05, compute_beat=True)
        if all_mt_feat.shape[0] == 0:
            print('Error: No data found! Check input folder')
            return
        names_category_toviz = [ntpath.basename(w).replace('.wav', '').split(' --- ')[0] for w in wav_files]
        names_to_viz = [ntpath.basename(w).replace('.wav', '') for w in wav_files]
        scaler = StandardScaler()
        F = scaler.fit_transform(all_mt_feat)
        K1 = 2
        K2 = 10
        if K1 > F.shape[0]:
            K1 = F.shape[0]
        if K2 > F.shape[0]:
            K2 = F.shape[0]
        pca1 = sklearn.decomposition.PCA(n_components=K1)
        pca1.fit(F)
        pca2 = sklearn.decomposition.PCA(n_components=K2)
        pca2.fit(F)
        finalDims = pca1.transform(F)
        finalDims2 = pca2.transform(F)
    else:
        (all_mt_feat, Ys, wav_files) = aF.directory_feature_extraction_no_avg(folder, 20.0, 5.0, 0.04, 0.04)
        if all_mt_feat.shape[0] == 0:
            print('Error: No data found! Check input folder')
            return
        names_category_toviz = [ntpath.basename(w).replace('.wav', '').split(' --- ')[0] for w in wav_files]
        names_to_viz = [ntpath.basename(w).replace('.wav', '') for w in wav_files]
        ldaLabels = Ys
        if priorKnowledge == 'artist':
            unames_category_toviz = list(set(names_category_toviz))
            YsNew = np.zeros(Ys.shape)
            for (i, uname) in enumerate(unames_category_toviz):
                indicesUCategories = [j for (j, x) in enumerate(names_category_toviz) if x == uname]
                for j in indicesUCategories:
                    indices = np.nonzero(Ys == j)
                    YsNew[indices] = i
            ldaLabels = YsNew
        scaler = StandardScaler()
        F = scaler.fit_transform(all_mt_feat)
        clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=10)
        clf.fit(F, ldaLabels)
        reducedDims = clf.transform(F)
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(reducedDims)
        reducedDims = pca.transform(reducedDims)
        uLabels = np.sort(np.unique(Ys))
        reducedDimsAvg = np.zeros((uLabels.shape[0], reducedDims.shape[1]))
        finalDims = np.zeros((uLabels.shape[0], 2))
        for (i, u) in enumerate(uLabels):
            indices = [j for (j, x) in enumerate(Ys) if x == u]
            f = reducedDims[indices, :]
            finalDims[i, :] = f.mean(axis=0)
        finalDims2 = reducedDims
    for i in range(finalDims.shape[0]):
        plt.text(finalDims[i, 0], finalDims[i, 1], ntpath.basename(wav_files[i].replace('.wav', '')), horizontalalignment='center', verticalalignment='center', fontsize=10)
        plt.plot(finalDims[i, 0], finalDims[i, 1], '*r')
    plt.xlim([1.2 * finalDims[:, 0].min(), 1.2 * finalDims[:, 0].max()])
    plt.ylim([1.2 * finalDims[:, 1].min(), 1.2 * finalDims[:, 1].max()])
    plt.show()
    SM = 1.0 - distance.squareform(distance.pdist(F, 'cosine'))
    unames_category_toviz = sort(list(set(names_category_toviz)))
    finalDimsGroup = np.zeros((len(unames_category_toviz), finalDims2.shape[1]))
    for (i, uname) in enumerate(unames_category_toviz):
        indices = [j for (j, x) in enumerate(names_category_toviz) if x == uname]
        f = finalDims2[indices, :]
        finalDimsGroup[i, :] = f.mean(axis=0)
    SMgroup = 1.0 - distance.squareform(distance.pdist(finalDimsGroup, 'cosine'))
    data = SMgroup
    fig = px.imshow(data, labels=dict(x='', y='', color='Category similarity'), x=unames_category_toviz, y=unames_category_toviz)
    fig.update_xaxes(side='top')
    fig.show()