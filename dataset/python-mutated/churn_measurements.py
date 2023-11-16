from __future__ import division
import numpy as np
__author__ = 'Eric Chiang'
__email__ = 'eric[at]yhathq.com'
'\n\nMeasurements inspired by Philip Tetlock\'s "Expert Political Judgment"\n\nEquations take from Yaniv, Yates, & Smith (1991):\n  "Measures of Descrimination Skill in Probabilistic Judgement"\n\n'

def calibration(prob, outcome, n_bins=10):
    if False:
        for i in range(10):
            print('nop')
    'Calibration measurement for a set of predictions.\n\n    When predicting events at a given probability, how far is frequency\n    of positive outcomes from that probability?\n    NOTE: Lower scores are better\n\n    prob: array_like, float\n        Probability estimates for a set of events\n\n    outcome: array_like, bool\n        If event predicted occurred\n\n    n_bins: int\n        Number of judgement categories to prefrom calculation over.\n        Prediction are binned based on probability, since "descrete" \n        probabilities aren\'t required. \n\n    '
    prob = np.array(prob)
    outcome = np.array(outcome)
    c = 0.0
    judgement_bins = np.arange(n_bins + 1) / n_bins
    bin_num = np.digitize(prob, judgement_bins)
    for j_bin in np.unique(bin_num):
        in_bin = bin_num == j_bin
        predicted_prob = np.mean(prob[in_bin])
        true_bin_prob = np.mean(outcome[in_bin])
        c += np.sum(in_bin) * (predicted_prob - true_bin_prob) ** 2
    return c / len(prob)

def discrimination(prob, outcome, n_bins=10):
    if False:
        while True:
            i = 10
    'Discrimination measurement for a set of predictions.\n\n    For each judgement category, how far from the base probability\n    is the true frequency of that bin?\n    NOTE: High scores are better\n\n    prob: array_like, float\n        Probability estimates for a set of events\n\n    outcome: array_like, bool\n        If event predicted occurred\n\n    n_bins: int\n        Number of judgement categories to prefrom calculation over.\n        Prediction are binned based on probability, since "descrete" \n        probabilities aren\'t required. \n\n    '
    prob = np.array(prob)
    outcome = np.array(outcome)
    d = 0.0
    base_prob = np.mean(outcome)
    judgement_bins = np.arange(n_bins + 1) / n_bins
    bin_num = np.digitize(prob, judgement_bins)
    for j_bin in np.unique(bin_num):
        in_bin = bin_num == j_bin
        true_bin_prob = np.mean(outcome[in_bin])
        d += np.sum(in_bin) * (true_bin_prob - base_prob) ** 2
    return d / len(prob)