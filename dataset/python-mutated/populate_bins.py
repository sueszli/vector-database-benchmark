"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.2

Populates the magnitude bins with the faults and scenarios that can generate these magnitudes.

@author: Thomas Chartier
"""
import numpy as np
import pickle

def pop(bin_mag, index_rup, rup_rates, M_min, re_use, f_bin_pop):
    if False:
        while True:
            i = 10
    if re_use == False:
        print('\t - Populating magnitude bins with ruptures')
        rupt_included = []
        for i_rup in index_rup:
            rupt_included_i = []
            involved_faults = rup_rates.get(str(i_rup)).get('involved_faults')
            for j_rup in index_rup:
                if set(rup_rates.get(str(j_rup)).get('involved_faults')) <= set(involved_faults):
                    mmax_j = rup_rates.get(str(j_rup)).get('Mmax')
                    rupt_included_i.append([j_rup, mmax_j])
            rupt_included.append(rupt_included_i)
        rup_in_bin = []
        for mag in bin_mag:
            rup_in_bin_i = []
            for i_rup in index_rup:
                involved_faults = rup_rates.get(str(i_rup)).get('involved_faults')
                Mmax = rup_rates.get(str(i_rup)).get('Mmax')
                if len(involved_faults) == 1:
                    if Mmax >= M_min and mag <= Mmax:
                        rup_in_bin_i.append(i_rup)
                elif mag <= Mmax:
                    add_scenario_to_bin = True
                    for j_rup in rupt_included[i_rup]:
                        if mag < j_rup[1] and j_rup[0] != i_rup:
                            add_scenario_to_bin = False
                    if add_scenario_to_bin == True:
                        rup_in_bin_i.append(i_rup)
            rup_in_bin.append(rup_in_bin_i)
        print('\t\t -> Bins populated.')
        with open(f_bin_pop, 'wb') as f:
            pickle.dump(rup_in_bin, f)
    else:
        print('Reloading bin pop from data file')
        with open(f_bin_pop, 'rb') as f:
            rup_in_bin = pickle.load(f)
    return rup_in_bin