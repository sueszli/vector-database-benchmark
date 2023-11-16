"""
This package contains a port of the knee-point detection package, kneed, by 
Kevin Arvai and hosted at https://github.com/arvkevi/kneed. This port is maintained 
with permission by the Yellowbrick contributors.
"""
import numpy as np
from scipy import interpolate
from scipy.signal import argrelextrema
import warnings
from yellowbrick.exceptions import YellowbrickWarning

class KneeLocator(object):
    """
    Finds the "elbow" or "knee" which is a value corresponding to the point of maximum curvature
    in an elbow curve, using knee point detection algorithm. This point is accessible via the
    `knee` attribute.

    Parameters
    ----------
    x : list
       A list of k values representing the no. of clusters in KMeans Clustering algorithm.

    y : list
       A list of k scores corresponding to each value of k. The type of k scores are determined 
       by the metric parameter from the KElbowVisualizer class.

    S : float, default: 1.0
       Sensitivity parameter that allows us to adjust how aggressive we want KneeLocator to
       be when detecting "knees" or "elbows".

    curve_nature : string, default: 'concave'
       A string that determines the nature of the elbow curve in which "knee" or "elbow" is
       to be found.

    curve_direction : string, default: 'increasing'
       A string that determines the increasing or decreasing nature of the elbow curve in
       which "knee" or "elbow" is to be found.

    online : bool, default: False
        kneed will correct old knee points if True, will return first knee if False

    Notes
    -----
    The KneeLocator is implemented using the "knee point detection algorithm" which can be read at
    `<https://www1.icsi.berkeley.edu/~barath/papers/kneedle-simplex11.pdf>`
    """

    def __init__(self, x, y, S=1.0, curve_nature='concave', curve_direction='increasing', online=False):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.array(x)
        self.y = np.array(y)
        self.curve_nature = curve_nature
        self.curve_direction = curve_direction
        self.N = len(self.x)
        self.S = S
        self.all_knees = set()
        self.all_norm_knees = set()
        self.all_knees_y = []
        self.all_norm_knees_y = []
        self.online = online
        uspline = interpolate.interp1d(self.x, self.y)
        self.Ds_y = uspline(self.x)
        self.x_normalized = self.__normalize(self.x)
        self.y_normalized = self.__normalize(self.Ds_y)
        self.y_normalized = self.transform_y(self.y_normalized, self.curve_direction, self.curve_nature)
        self.y_difference = self.y_normalized - self.x_normalized
        self.x_difference = self.x_normalized.copy()
        self.maxima_indices = argrelextrema(self.y_difference, np.greater_equal)[0]
        self.x_difference_maxima = self.x_difference[self.maxima_indices]
        self.y_difference_maxima = self.y_difference[self.maxima_indices]
        self.minima_indices = argrelextrema(self.y_difference, np.less_equal)[0]
        self.x_difference_minima = self.x_difference[self.minima_indices]
        self.y_difference_minima = self.y_difference[self.minima_indices]
        self.Tmx = self.y_difference_maxima - self.S * np.abs(np.diff(self.x_normalized).mean())
        (self.knee, self.norm_knee) = self.find_knee()
        self.knee_y = self.norm_knee_y = None
        if self.knee:
            self.knee_y = self.y[self.x == self.knee][0]
            self.norm_knee_y = self.y_normalized[self.x_normalized == self.norm_knee][0]
        if (self.all_knees or self.all_norm_knees) == set():
            warning_message = "No 'knee' or 'elbow point' detected This could be due to bad clustering, no actual clusters being formed etc."
            warnings.warn(warning_message, YellowbrickWarning)
            self.knee = None
            self.norm_knee = None
            self.knee_y = None
            self.norm_knee_y = None

    @staticmethod
    def __normalize(a):
        if False:
            for i in range(10):
                print('nop')
        '\n        Normalizes an array.\n        Parameters\n        -----------\n        a : list\n           The array to normalize\n        '
        return (a - min(a)) / (max(a) - min(a))

    @staticmethod
    def transform_y(y, direction, curve):
        if False:
            while True:
                i = 10
        'transform y to concave, increasing based on given direction and curve'
        if direction == 'decreasing':
            if curve == 'concave':
                y = np.flip(y)
            elif curve == 'convex':
                y = y.max() - y
        elif direction == 'increasing' and curve == 'convex':
            y = np.flip(y.max() - y)
        return y

    def find_knee(self):
        if False:
            return 10
        'This function finds and sets the knee value and the normalized knee value. '
        if not self.maxima_indices.size:
            warning_message = 'No "knee" or "elbow point" detected This could be due to bad clustering, no actual clusters being formed etc.'
            warnings.warn(warning_message, YellowbrickWarning)
            return (None, None)
        maxima_threshold_index = 0
        minima_threshold_index = 0
        for (i, x) in enumerate(self.x_difference):
            if i < self.maxima_indices[0]:
                continue
            j = i + 1
            if x == 1.0:
                break
            if (self.maxima_indices == i).any():
                threshold = self.Tmx[maxima_threshold_index]
                threshold_index = i
                maxima_threshold_index += 1
            if (self.minima_indices == i).any():
                threshold = 0.0
                minima_threshold_index += 1
            if self.y_difference[j] < threshold:
                if self.curve_nature == 'convex':
                    if self.curve_direction == 'decreasing':
                        knee = self.x[threshold_index]
                        norm_knee = self.x_normalized[threshold_index]
                    else:
                        knee = self.x[-(threshold_index + 1)]
                        norm_knee = self.x_normalized[threshold_index]
                elif self.curve_nature == 'concave':
                    if self.curve_direction == 'decreasing':
                        knee = self.x[-(threshold_index + 1)]
                        norm_knee = self.x_normalized[threshold_index]
                    else:
                        knee = self.x[threshold_index]
                        norm_knee = self.x_normalized[threshold_index]
                y_at_knee = self.y[self.x == knee][0]
                y_norm_at_knee = self.y_normalized[self.x_normalized == norm_knee][0]
                if knee not in self.all_knees:
                    self.all_knees_y.append(y_at_knee)
                    self.all_norm_knees_y.append(y_norm_at_knee)
                self.all_knees.add(knee)
                self.all_norm_knees.add(norm_knee)
                if self.online is False:
                    return (knee, norm_knee)
        if self.all_knees == set():
            return (None, None)
        return (knee, norm_knee)

    def plot_knee_normalized(self):
        if False:
            i = 10
            return i + 15
        '\n        Plots the normalized curve, the distance curve (x_distance, y_normalized) and the\n        knee, if it exists.\n        '
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.plot(self.x_normalized, self.y_normalized)
        plt.plot(self.x_difference, self.y_difference, 'r')
        plt.xticks(np.arange(self.x_normalized.min(), self.x_normalized.max() + 0.1, 0.1))
        plt.yticks(np.arange(self.y_difference.min(), self.y_normalized.max() + 0.1, 0.1))
        plt.vlines(self.norm_knee, plt.ylim()[0], plt.ylim()[1])

    def plot_knee(self):
        if False:
            while True:
                i = 10
        '\n        Plot the curve and the knee, if it exists\n\n        '
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.plot(self.x, self.y)
        plt.vlines(self.knee, plt.ylim()[0], plt.ylim()[1])

    @property
    def elbow(self):
        if False:
            return 10
        return self.knee

    @property
    def norm_elbow(self):
        if False:
            while True:
                i = 10
        return self.norm_knee

    @property
    def elbow_y(self):
        if False:
            print('Hello World!')
        return self.knee_y

    @property
    def norm_elbow_y(self):
        if False:
            return 10
        return self.norm_knee_y

    @property
    def all_elbows(self):
        if False:
            for i in range(10):
                print('nop')
        return self.all_knees

    @property
    def all_norm_elbows(self):
        if False:
            i = 10
            return i + 15
        return self.all_norm_knees

    @property
    def all_elbows_y(self):
        if False:
            return 10
        return self.all_knees_y

    @property
    def all_norm_elbows_y(self):
        if False:
            while True:
                i = 10
        return self.all_norm_knees_y