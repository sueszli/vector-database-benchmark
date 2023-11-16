"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.2

@author: Thomas Chartier
"""
import numpy as np

def get_rate_model(rup_rates, fault_prop, bin_mag):
    if False:
        for i in range(10):
            print('nop')
    rate_f_in_model = []
    for dict in rup_rates:
        rate_f_in_model.append(rup_rates.get(dict).get('rates'))
    rate_f_in_model = np.sum(rate_f_in_model, axis=0)
    rate_bg_in_model = np.zeros(len(bin_mag))
    for index_mag in range(len(bin_mag)):
        rate_bg_in_model[index_mag] += (1 - fault_prop(bin_mag[index_mag])) * rate_f_in_model[index_mag] / fault_prop(bin_mag[index_mag])
    rate_tot_model = rate_f_in_model + rate_bg_in_model
    return rate_tot_model

def get_rate_faults_n_bg(rup_rates, fault_prop, bin_mag):
    if False:
        i = 10
        return i + 15
    rate_f_in_model = []
    for dict in rup_rates:
        rate_f_in_model.append(rup_rates.get(dict).get('rates'))
    rate_f_in_model = np.sum(rate_f_in_model, axis=0)
    rate_bg_in_model = np.zeros(len(bin_mag))
    for index_mag in range(len(bin_mag)):
        rate_bg_in_model[index_mag] += (1 - fault_prop(bin_mag[index_mag])) * rate_f_in_model[index_mag] / fault_prop(bin_mag[index_mag])
    return (rate_f_in_model, rate_bg_in_model)