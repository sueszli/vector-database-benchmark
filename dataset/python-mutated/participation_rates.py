"""
This modules exctracts the participation rate for a given fault
"""
import pickle, csv
import matplotlib.pyplot as plt
import numpy as np

def get_all_participation_rates(data, param, fault_names):
    if False:
        return 10
    '\n    data : list (MFDs_to_pkl in main code)\n    '
    fault_names = data[1]
    mfd_singles = data[2]
    mfd_total = [0 for _ in mfd_singles[0]]
    for i in range(len(mfd_singles)):
        mfd_total = [x + y for (x, y) in zip(mfd_total, mfd_singles[i])]
    mfd_multi = data[4]
    for i in range(len(mfd_multi)):
        mfd_total = [x + y for (x, y) in zip(mfd_total, mfd_multi[i])]
    involved_faults = data[5]
    ruptures = {}
    with open(param['dirpath'] + param['main']['rupture_file'], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        next(spamreader)
        rup_id = 0
        for row in spamreader:
            ruptures.update({rup_id: [str(i) for i in row[0].split(' ')[:-1]]})
            rup_id += 1
    "\n    # get the properties of the faults\n    prop_file = open(path + '../../LOG/ModelMultiFaultsTest_prop.pkl','rb')\n    #prop_file = open(path + '../../LOG/DOM_v1_prop.pkl','rb')\n    f_prop = pickle.load(prop_file)\n    "
    dict_p_rates = {}
    all_non_zero_rups = []
    for fault_name in fault_names:
        "\n        fault_id = None\n        for fid in f_prop.keys():\n            if f_prop[fid]['name'] == fault_name :\n                fault_id = fid\n        "
        fault_id = fault_names.index(fault_name)
        p_rate = [0 for _ in mfd_total]
        p_rate = [x + y for (x, y) in zip(p_rate, mfd_singles[fault_id])]
        i_sc = 0
        non_zero_rups = 0
        for involved_faults_i in involved_faults:
            if fault_id in list(involved_faults_i[0]):
                p_rate = [sum(x) for x in zip(p_rate, mfd_multi[i_sc])]
                if sum(mfd_multi[i_sc]) != 0.0:
                    non_zero_rups += 1
            i_sc += 1
        dict_p_rates.update({fault_name: p_rate})
        all_non_zero_rups.append(non_zero_rups)
    return (dict_p_rates, mfd_total, all_non_zero_rups)

def get_bin_mag(mfd_total, Mmin):
    if False:
        for i in range(10):
            print('nop')
    bin_mag = []
    for i in range(len(mfd_total)):
        bin_mag.append(Mmin + 0.1 * i)
    return bin_mag

def extract_rates(fault_name, dict_p_rates):
    if False:
        for i in range(10):
            print('nop')
    incremental_rate = dict_p_rates[fault_name]
    cumulative_rate = [sum(incremental_rate[i:]) for i in range(len(incremental_rate))]
    return (incremental_rate, cumulative_rate)

def plot_participation_rates(bin_mag, incremental_rate, cumulative_rate, fault_name, ptf):
    if False:
        i = 10
        return i + 15
    '\n    ptf : path to fig\n    '
    plt.scatter(bin_mag, incremental_rate, c='k', label='incremental rates')
    plt.plot(bin_mag, cumulative_rate, c='k', label='cumulative rates')
    plt.title(fault_name)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('magnitude')
    plt.ylabel('annual earthquake rate')
    plt.gcf().set_dpi(120)
    plt.savefig(ptf)
    plt.close()