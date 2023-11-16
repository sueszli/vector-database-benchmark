"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.2

@author: Thomas Chartier
"""
import numpy as np

def select(sr_values, sample, index_fault, linked_level, list_quater_picked):
    if False:
        return 10
    sr_min = sr_values[0]
    sr_mean = sr_values[1]
    sr_max = sr_values[2]
    if sample == 1:
        slip_rate = sr_mean
    elif sum(linked_level) == 0:
        slip_rate_inf = np.random.uniform(sr_min, sr_mean)
        slip_rate_sup = np.random.uniform(sr_mean, sr_max)
        slip_rate = np.random.choice([slip_rate_inf, slip_rate_sup])
    else:
        value_lvl = 10
        quarters_picked = []
        for index_c in range(len(linked_level)):
            if linked_level[index_c] != 0:
                if list_quater_picked[index_c] != 0:
                    if linked_level[index_c] < value_lvl:
                        quarters_picked = []
                        value_lvl = linked_level[index_c]
                    if linked_level[index_c] == value_lvl:
                        quarters_picked.append(list_quater_picked[index_c])
        if quarters_picked == []:
            slip_rate_inf = np.random.uniform(sr_min, sr_mean)
            slip_rate_sup = np.random.uniform(sr_mean, sr_max)
            slip_rate = np.random.choice([slip_rate_inf, slip_rate_sup])
            if slip_rate < sr_min + 1.0 / 2.0 * (sr_mean - sr_min):
                quarter_to_pick = 1
            elif slip_rate < sr_mean:
                quarter_to_pick = 2
            elif slip_rate < sr_mean + 1.0 / 2.0 * (sr_max - sr_mean):
                quarter_to_pick = 3
            else:
                quarter_to_pick = 4
        else:
            quarter_to_pick = max(set(quarters_picked), key=quarters_picked.count)
            if quarter_to_pick == 1:
                slip_rate = np.random.uniform(sr_min, sr_min + 1.0 / 2.0 * (sr_mean - sr_min) + 1e-06)
            elif quarter_to_pick == 2:
                slip_rate = np.random.uniform(sr_min + 1.0 / 2.0 * (sr_mean - sr_min), sr_mean + 1e-06)
            elif quarter_to_pick == 3:
                slip_rate = np.random.uniform(sr_mean, sr_mean + 1.0 / 2.0 * (sr_max - sr_mean) + 1e-06)
            elif quarter_to_pick == 4:
                slip_rate = np.random.uniform(sr_mean + 1.0 / 2.0 * (sr_max - sr_mean), sr_max + 1e-06)
        list_quater_picked[index_fault] = quarter_to_pick
    return slip_rate