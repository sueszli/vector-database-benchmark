"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.3

This code is pretty much the core of SHERIFS. It converts the slip-rate into earthquake rates.

@author: Thomas Chartier
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import scipy
import math
from scipy.interpolate import interp1d
import scalling_laws
import populate_bins
import mfd_shape
import time
import warnings
import core_utils, target, rates
from seismic_moment import mag_to_M0
warnings.simplefilter('ignore', RuntimeWarning)

class EQ_on_faults_from_sr:

    def __init__(self, Run_Name, M_min, mfd_param, faults_names, faults_area, faults_length, faults_width, faults_slip_rates, scenarios, faults_shear_mod, path, pathlog, sample, selected_ScL, dimention_used, use_all_ScL_data, faults_mecanism, bg_ratio, size_of_increment, mfd_hyp, count_reruns, faults_lon, faults_lat, Mmax_range, calculation_log_file, branch, param):
        if False:
            return 10
        self.Run_Name = Run_Name
        self.M_min = M_min
        self.mfd_param = mfd_param
        self.faults_names = faults_names
        self.faults_area = faults_area
        self.faults_length = faults_length
        self.faults_width = faults_width
        self.faults_slip_rates = faults_slip_rates
        self.scenarios = scenarios
        self.faults_shear_mod = faults_shear_mod
        self.path = path
        self.pathlog = pathlog
        self.sample = sample
        self.selected_ScL = selected_ScL
        self.dimention_used = dimention_used
        self.use_all_ScL_data = use_all_ScL_data
        self.faults_mecanism = faults_mecanism
        self.bg_ratio = bg_ratio
        self.size_of_increment = size_of_increment
        self.mfd_hyp = mfd_hyp
        self.count_reruns = count_reruns
        self.faults_lon = faults_lon
        self.faults_lat = faults_lat
        self.Mmax_range = Mmax_range
        self.calculation_log_file = calculation_log_file
        self.branch = branch
        self.param = param
        self.initialize()

    def initialize(self):
        if False:
            print('Hello World!')
        faults_shear_mod = self.faults_shear_mod
        mfd_param = self.mfd_param
        M_min = self.M_min
        Mmax_range = self.Mmax_range
        inti_core_time = time.time()
        faults_names = self.faults_names
        faults_areas = self.faults_area
        faults_length = self.faults_length
        faults_width = self.faults_width
        faults_slip_rates = self.faults_slip_rates
        log_calculation_file = open(self.pathlog + '/calculation_sample_' + str(self.sample) + '.txt', 'w')
        log_sliprep_file = self.pathlog + '/sliprep_sample_' + str(self.sample) + '.pkl'
        re_use = True
        scenarios_names = self.scenarios
        run_name = self.Run_Name
        model_name = self.branch['model']
        scl_name = self.branch['scl'][0] + '_' + self.branch['scl'][1] + '_' + self.branch['scl'][2]
        set_name = self.branch['set']
        f_mmax = self.param['dirpath'] + run_name + '/LOG/' + model_name + '_mmax_' + scl_name + '_' + set_name + '_' + str(self.sample) + '.pkl'
        if not os.path.isfile(f_mmax):
            re_use = False
        if re_use == False:
            print('Building scenarios and Mmax')
            '##################################################################\n            # finds the faults belonging to each scenario\n            ##################################################################'
            index_faults_in_scenario = []
            for scenario in scenarios_names:
                i_scenario = []
                for i in range(len(scenario)):
                    i_scenario = i_scenario + list(np.where(np.array(faults_names) == scenario['f_%s' % str(i + 1)]))
                i_scenario = np.reshape(i_scenario, (1, len(scenario)))
                index_faults_in_scenario.append(i_scenario)
            self.index_faults_in_scenario = index_faults_in_scenario
            faults_alone = []
            faults_isolated = []
            len_max_section_scenario = 0
            for indexes in index_faults_in_scenario:
                if len(indexes[0]) > len_max_section_scenario:
                    len_max_section_scenario = len(indexes[0])
            index_fault = 0
            for fault_name in faults_names:
                fault_alone_bool = True
                fault_isolated_bool = True
                for indexes in index_faults_in_scenario:
                    if index_fault in indexes[0]:
                        fault_alone_bool = False
                        if len(indexes[0]) > len_max_section_scenario / 2.0:
                            fault_isolated_bool = False
                if fault_isolated_bool == True and fault_alone_bool == False:
                    faults_isolated.append(index_fault)
                if fault_alone_bool == True:
                    faults_alone.append(index_fault)
                index_fault += 1
            '##################################################################\n            # area and length of each scenario\n            ##################################################################'
            scenario_area = []
            index_faults_in_scenario_for_scl = []
            for indexes_i in index_faults_in_scenario:
                list_of_points = []
                index_for_scenario = []
                for index_fault in indexes_i[0]:
                    list_i = str(self.faults_lon[index_fault][0]) + str(self.faults_lon[index_fault][-1]) + str(self.faults_lat[index_fault][0]) + str(self.faults_lat[index_fault][-1])
                    if not list_i in list_of_points:
                        list_of_points.append(list_i)
                        index_for_scenario.append(index_fault)
                index_faults_in_scenario_for_scl.append(index_for_scenario)
            for i in index_faults_in_scenario_for_scl:
                scenario_i_area = np.sum(np.take(faults_areas, i))
                scenario_area.append(scenario_i_area)
            scenario_length = []
            for i in index_faults_in_scenario_for_scl:
                scenario_i_length = np.sum(np.take(faults_length, i))
                scenario_length.append(scenario_i_length)
            scenario_width = []
            for i in index_faults_in_scenario_for_scl:
                scenario_i_width = np.mean(np.take(faults_width, i))
                scenario_width.append(scenario_i_width)
            '\n            check the max dimention for a single fault according to the\n            aspect ratio\n            '
            (f_len_ar, f_width_ar, f_area_ar) = ([], [], [])
            ar = self.param['main']['parameters']['aspect_ratio']
            for (l, w) in zip(self.faults_length, self.faults_width):
                if l < w * ar:
                    f_len_ar.append(l)
                    f_width_ar.append(l * ar)
                    f_area_ar.append(l * l * ar)
                else:
                    f_len_ar.append(l)
                    f_width_ar.append(w)
                    f_area_ar.append(l * w)
            '##################################################################\n            #####################################################################\n            #\n            # setting up the coefficients for the scalling law\n            # and calculate the Mmax\n            #\n            #####################################################################\n            ##################################################################'
            Mmaxmin = Mmax_range[0]
            Mmaxmax = Mmax_range[1]
            Mmax = -1.0
            loop_Mmax = 1
            while Mmax < Mmaxmin or Mmax > Mmaxmax:
                log_Mmax_file = open(self.param['dirpath'] + run_name + '/LOG/' + model_name + '_Log_Mmax_sample_' + scl_name + '_' + set_name + '_' + str(self.sample) + '.txt', 'w')
                if loop_Mmax == 1:
                    Mmaxs = scalling_laws.Calc_Mmax(f_area_ar, scenario_area, f_len_ar, scenario_length, f_width_ar, scenario_width, self.selected_ScL, self.dimention_used, self.use_all_ScL_data, self.faults_mecanism, index_faults_in_scenario, self.sample)
                else:
                    Mmaxs = scalling_laws.Calc_Mmax(f_area_ar, scenario_area, f_len_ar, scenario_length, f_width_ar, scenario_width, self.selected_ScL, self.dimention_used, self.use_all_ScL_data, self.faults_mecanism, index_faults_in_scenario, 10)
                Mmax_faults = Mmaxs.Mmax_faults
                Mmax_scenario = Mmaxs.Mmax_scenario
                final_fault_length = Mmaxs.final_fault_length
                final_scenario_length = Mmaxs.final_scenario_length
                index_fault = 0
                for Mmax_i in Mmax_faults:
                    line = faults_names[index_fault] + '\t' + str(round(faults_length[index_fault] / 1000.0, 1)) + '\t' + final_fault_length[index_fault] + '\t' + str(round(self.faults_width[index_fault], 1)) + '\t' + str(round(self.faults_area[index_fault] / 1000000.0, 1)) + '\t' + str(Mmax_i) + '\n'
                    log_Mmax_file.write(line)
                    index_fault += 1
                index_scenario = 0
                for Mmax_i in Mmax_scenario:
                    line = str(scenarios_names[index_scenario]) + '\t' + str(round(scenario_length[index_scenario] / 1000.0, 1)) + '\t' + final_scenario_length[index_scenario] + '\t' + str(round(scenario_width[index_scenario], 1)) + '\t' + str(round(scenario_area[index_scenario] / 1000000.0, 1)) + '\t' + str(Mmax_i) + '\n'
                    log_Mmax_file.write(line)
                    index_scenario += 1
                log_Mmax_file.close()
                if np.size(scenarios_names) == 0:
                    Mmax = max(Mmax_faults)
                else:
                    Mmax = max((max(Mmax_faults), max(Mmax_scenario)))
                loop_Mmax += 1
                if loop_Mmax == 30:
                    print('An Mmax incompatible with the ruptures is imposed!! Change it in run.info or change the rupture.txt file')
            if loop_Mmax >= 3:
                print('Mmax imposed: ' + str(Mmax))
                self.calculation_log_file.write('\nMmax imposed: ' + str(Mmax))
            with open(f_mmax, 'wb') as f:
                dump_to_file = [Mmax_faults, Mmax_scenario, final_fault_length, final_scenario_length, faults_alone, faults_isolated, index_faults_in_scenario, scenario_area]
                pickle.dump(dump_to_file, f)
            print('\t - scenario and max built')
        else:
            print('Reloading Mmax from data file')
            with open(f_mmax, 'rb') as f:
                load_from_file = pickle.load(f)
                Mmax_faults = load_from_file[0]
                Mmax_scenario = load_from_file[1]
                final_fault_length = load_from_file[2]
                final_scenario_length = load_from_file[3]
                faults_alone = load_from_file[4]
                faults_isolated = load_from_file[5]
                index_faults_in_scenario = load_from_file[6]
                scenario_area = load_from_file[7]
                if np.size(scenarios_names) == 0:
                    Mmax = max(Mmax_faults)
                else:
                    Mmax = max((max(Mmax_faults), max(Mmax_scenario)))
                print('\t - scenario and max loaded')
        '##################################################################\n        #etablish the magnitude bin (0.1)\n        ##################################################################'
        bin_mag = np.linspace(M_min, Mmax, int(round(Mmax - M_min, 1) * 10.0 + 1))
        '#####################################################################\n        # initializing incremental reccurence of the earthquakes for each rupture and each bin)\n        #####################################################################'
        rup_rates = {}
        index_rup = []
        for i in range(len(faults_names)):
            rup_rates.update({str(i): {'rup_id': i, 'rup_name': faults_names[i], 'involved_faults': [i], 'Mmax': Mmax_faults[i], 'area': faults_areas[i], 'rates': np.zeros(len(bin_mag))}})
            index_rup.append(i)
        i_end = i + 1
        if np.size(scenarios_names) != 0:
            for j in range(len(scenarios_names)):
                rup_rates.update({str(i_end + j): {'rup_id': i_end + j, 'rup_name': scenarios_names[j], 'involved_faults': index_faults_in_scenario[j][0], 'Mmax': Mmax_scenario[j], 'area': scenario_area[j], 'rates': np.zeros(len(bin_mag))}})
                index_rup.append(i_end + j)
        if str(self.sample) == '1':
            log_rup_file = open(self.pathlog + '/ruptures.txt', 'w')
            log_rup_file.write('rup_id\tinvolved_faults\n')
            for i in range(len(rup_rates)):
                log_rup_file.write(str(rup_rates.get(str(i)).get('rup_id')) + '\t')
                for j in rup_rates.get(str(i)).get('involved_faults'):
                    log_rup_file.write(str(j) + ' ')
                log_rup_file.write('\n')
            log_rup_file.close()
        self.index_faults_in_scenario = index_faults_in_scenario
        '#####################################################################\n        # For each bin, find which fault and which scenario populates it.\n        #####################################################################'
        f_bin_pop = self.param['dirpath'] + run_name + '/LOG/' + model_name + '_bin_pop_' + scl_name + '_' + set_name + '_' + str(self.sample) + '.pkl'
        if not os.path.isfile(f_bin_pop):
            re_use = False
        rup_in_bin = populate_bins.pop(bin_mag, index_rup, rup_rates, M_min, re_use, f_bin_pop)
        '##################################################################\n        # Definition of the background - how much is not on the faults\n        ##################################################################'
        bin_mag_fault_prop = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
        fault_prop_inc = self.bg_ratio
        bin_mag_fault_prop.append(10.0)
        fault_prop_inc = np.append(np.array(fault_prop_inc), 1.0)
        fault_prop = interp1d(bin_mag_fault_prop, fault_prop_inc)
        self.fault_prop = fault_prop
        '##################################################################\n        #Calculation of the total moment rate sum(slip-rate)*sum(area) + moment in the background\n        ##################################################################'
        Total_moment_faults_rate_init = 0.0
        index_fault = 0
        for fault_area in faults_areas:
            Total_moment_faults_rate_init += faults_shear_mod[index_fault] * fault_area * faults_slip_rates[index_fault]
            index_fault += 1
        log_calculation_file.write('Moment_rate_fault_initial calculated using muAs' + '\t' + str(Total_moment_faults_rate_init) + '\t' + 'N.m' + '\n')
        Total_moment_rate_fault = 0.0
        '##################################################################\n        #etablish the target regional distribution\n        ##################################################################'
        if self.mfd_hyp == 'GR':
            p_MFD = mfd_shape.GR(mfd_param, bin_mag)
        elif self.mfd_hyp == 'tapered_GR':
            mfd_param['M_corner'] = Mmax - 0.4
            p_MFD = mfd_shape.tapered_GR(mfd_param, bin_mag)
        elif self.mfd_hyp == 'double_GR':
            p_MFD = mfd_shape.double_GR(mfd_param, bin_mag)
        elif self.mfd_hyp == 'YC':
            p_MFD = mfd_shape.YC(mfd_param, bin_mag, Mmax)
        elif self.mfd_hyp == 'YC_marmara':
            p_MFD = mfd_shape.YC_marmara(mfd_param, bin_mag, Mmax)
        elif self.mfd_hyp == 'YC_modified':
            p_MFD = mfd_shape.YC_modified(mfd_param, bin_mag, Mmax)
        elif self.mfd_hyp == 'UCERF_DV':
            p_MFD = mfd_shape.UCERF_DV(bin_mag)
        else:
            print('Error : Unknown MFD name!')
        p_MFD = p_MFD / sum(p_MFD)
        '##################################################################\n        # Converting in term of moment\n        ##################################################################'
        p_MFD_MO = []
        index_mag = 0
        for mag in bin_mag:
            M0 = mag_to_M0(mag)
            p_MFD_MO.append(fault_prop(mag) * p_MFD[index_mag] * M0)
            index_mag += 1
        p_MFD_MO = p_MFD_MO / sum(p_MFD_MO)
        '##################################################################\n        # Adding the moment rate of the background to the Total_moment_rate_init\n        ##################################################################'
        Total_moment_rate_init = Total_moment_faults_rate_init
        index_mag = 0
        for mag in bin_mag:
            moment_rate_fault_bin = p_MFD_MO[index_mag] * Total_moment_faults_rate_init
            moment_rate_BG_bin = (1 - fault_prop(mag)) * moment_rate_fault_bin / fault_prop(mag)
            Total_moment_rate_init += moment_rate_BG_bin
            index_mag += 1
        '##################################################################\n        # Etablish the target of moment rate per bin\n        ##################################################################'
        target_moment_per_bin = p_MFD_MO * Total_moment_faults_rate_init
        '##################################################################\n        # Create the sliprate budget dictionary\n        # This matrix contain a list of the fault name. each name is repeted\n        # by a number of time depending of the slip-rate of the fault.\n        ##################################################################'
        size_of_increment = float(self.size_of_increment) * 0.001
        if self.count_reruns != 1:
            size_of_increment = size_of_increment / (float(self.count_reruns) * 1.5 - 1.0)
        faults_budget = {}
        for index_fault in range(len(faults_names)):
            nb_dsr = int(round(faults_slip_rates[index_fault] / size_of_increment, 0))
            if nb_dsr == 0:
                nb_dsr = 1
            faults_budget.update({index_fault: nb_dsr})
        min_budget = float(min(faults_budget.values())) + 1
        max_budget = float(max(faults_budget.values()))
        while max_budget / min_budget > 50.0:
            min_budget *= 2.0
        '##################################################################\n        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #\n        #   Populate the bin of magnitude of each fault and each scenario\n        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #\n        ##################################################################'
        M_slip_repartition = {}
        for (fault, i) in zip(faults_names, range(len(faults_names))):
            dic_tmp = {}
            for rup_i in range(len(rup_rates)):
                if i in rup_rates.get(str(rup_i)).get('involved_faults'):
                    dic_tmp.update({str(rup_i): 0})
            dic_tmp.update({'NMS': 0})
            M_slip_repartition.update({str(fault): dic_tmp})
        moment_rate_in_bin = np.zeros(len(bin_mag))
        model_MFD = []
        number_of_loops = 0
        number_of_loops_before = 0
        number_of_loops_for_nothing = 0
        number_of_loops_for_nothing_before = 0
        number_of_loops_last_checked = -1
        empty_bins = []
        empty_rups = []
        bin_target_reached = []
        len_faults_budget = []
        aseismic_count = 0
        color_mag = []
        '######################\n        #####   MAIN LOOP   #####\n        ######################'
        TARGET = []
        nb_ss_to_spend = float(sum(faults_budget.values()))
        sum_fault_budget = nb_ss_to_spend
        print('Number of dsr to spend : ' + str(nb_ss_to_spend))
        print('Min of sdr :', min(faults_budget.values()))
        print('Max of sdr :', max(faults_budget.values()))
        self.calculation_log_file.write('\nnumber of dsr to spend : ' + str(nb_ss_to_spend) + '\n')
        print_percent = True
        do_the_target = True
        bool_target_set = False
        uniform_spending = self.param['main']['parameters']['uniform_spending']
        if uniform_spending in ['True', 'true']:
            uniform_spending = True
        else:
            uniform_spending = False
        deep_analysis = self.param['main']['parameters']['deep_analysis']
        if deep_analysis in ['True', 'true']:
            deep_analysis = True
        else:
            deep_analysis = False
        faster_rup_weight = self.param['main']['parameters']['faster_rup_weight']
        if faster_rup_weight in ['True', 'true']:
            faster_rup_weight = True
        else:
            faster_rup_weight = False
        option_fast = self.param['main']['parameters']['option_fast']
        if option_fast in ['True', 'true']:
            option_fast = True
        else:
            option_fast = False
        local_MFD = self.param['main']['parameters']['local_MFD']
        if local_MFD in ['True', 'true']:
            f_mfd_area = self.param['main']['local_MFD_file']
            (local_mfds, associated_rup, associated_weight) = core_utils.link_rup_mfd_area(rup_rates, f_mfd_area, self.faults_lon, self.faults_lat, bin_mag, self.bg_ratio)
        time_weight_rupt = 0.0
        time_clean_w_rupt = 0.0
        time_target_building = 0.0
        time_checking_target_reach = 0.0
        time_spending_dsr = 0.0
        time_checking_empty_bin = 0.0
        time_checking_empty_faults = 0.0
        budget_init = int(sum_fault_budget)
        if faster_rup_weight == True:
            nb_weigthings_rup_sampling = int(self.param['main']['parameters']['nb_weigthings_rup_sampling'])
            if 'type_weigthings_rup_sampling' in self.param['main']['parameters'].keys():
                type_weigthings_rup_sampling = self.param['main']['parameters']['type_weigthings_rup_sampling']
            else:
                type_weigthings_rup_sampling = 'lin'
            if type_weigthings_rup_sampling == 'log':
                weigthing_built = [int(i) for i in np.logspace(0.0, np.log10(budget_init), nb_weigthings_rup_sampling)]
            if type_weigthings_rup_sampling == 'lin':
                weigthing_built = [int(i) for i in np.linspace(0.0, budget_init, nb_weigthings_rup_sampling)]
        else:
            weigthing_built = [int(i) for i in range(budget_init)]
        weigthing_built.reverse()
        weigth_rup_sample = 0
        slip_rate_use_per_fault = np.zeros(len(faults_names))
        moment_rate_required = 0.0
        moment_rate_left = Total_moment_faults_rate_init
        rate_tot_model = rates.get_rate_model(rup_rates, fault_prop, bin_mag)
        rate_in_model = np.zeros(len(bin_mag))
        test_mean_picked = []
        most_likely_pick = []
        n_w_work = 0
        n_w_crash = 0
        loop_last_rup_w = 0
        picked_empty_rup = 0
        old_percent = '0000'
        while sum_fault_budget > 0:
            ratio_done = 1.0 - float(sum_fault_budget) / nb_ss_to_spend
            if ratio_done > 0.01:
                (model_MFD, self.calculation_log_file, print_percent) = core_utils.progress(model_MFD, self.calculation_log_file, ratio_done, print_percent, rup_rates, fault_prop, bin_mag)
            number_of_loops += 1
            if len(empty_bins) != len(bin_mag):
                ' Calculate the new target shape in each bin in terms of moment rate '
                tmp = time.time()
                target_i = target.get_new_target(number_of_loops, moment_rate_in_bin, p_MFD_MO, target_moment_per_bin, bin_mag, empty_bins, bin_target_reached, rup_in_bin)
                time_target_building += time.time() - tmp
                if sum(target_i) == 0.0:
                    target_i = p_MFD_MO
                try:
                    most_likely_pick.append(bin_mag[list(target_i).index(max(list(target_i)))])
                except:
                    print(target_i)
                    most_likely_pick.append(bin_mag[list(target_i).index(max(list(target_i)))])
                if deep_analysis == True:
                    percent = round((1.0 - float(sum_fault_budget) / float(nb_ss_to_spend)) * 100.0)
                    percent = '{:04d}'.format(percent)
                    if percent != old_percent:
                        old_percent = percent
                        print('\nnumber_of_loops', number_of_loops)
                        print('budget left : ', sum_fault_budget, ' | ', percent, '%')
                        time_str = core_utils.seconds_to_str(time_target_building)
                        print('time building target at time i : ', time_str)
                        time_str = core_utils.seconds_to_str(time_weight_rupt)
                        print('time weighting rupture pick : ', time_str)
                        time_str = core_utils.seconds_to_str(time_clean_w_rupt)
                        print('time cleaning weighting rupture weigth : ', time_str)
                        time_str = core_utils.seconds_to_str(time_checking_target_reach)
                        print('time checking target reach : ', time_str)
                        time_str = core_utils.seconds_to_str(time_checking_empty_bin)
                        print('time checking empty bins : ', time_str)
                        time_str = core_utils.seconds_to_str(time_checking_empty_faults)
                        print('time checking empty faults : ', time_str)
                        time_str = core_utils.seconds_to_str(time_target_building)
                        print('time spending dsr : ', time_str)
                        time_str = core_utils.seconds_to_str(time.time() - inti_core_time)
                        print('total core time : ', time_str)
                        tot_core_time = time.time() - inti_core_time
                        unaccounted_t = tot_core_time - (time_target_building + time_weight_rupt + time_clean_w_rupt + time_checking_target_reach + time_checking_empty_bin + time_checking_empty_faults + time_spending_dsr)
                        time_str = core_utils.seconds_to_str(unaccounted_t)
                        print('unaccounted time : ', time_str)
                        print('max target : ', round(max(target_i), 4), '| last bin w : ', str(round(target_i[-1], 6)))
                        print('Empty mag bins:')
                        print(empty_bins)
                        budget_last_bin = 0
                        for rup_i in rup_in_bin[-1]:
                            if not str(rup_i) in empty_rups:
                                for f_i in rup_rates.get(str(rup_i)).get('involved_faults'):
                                    budget_last_bin += faults_budget[f_i]
                        print('fault budget last bin :', budget_last_bin)
                        (fig, (ax0, ax1)) = plt.subplots(ncols=2)
                        ax0.plot(bin_mag, target_i, label='target_i', marker='x')
                        ax0.plot(bin_mag, p_MFD_MO, label='p_MFD_MO', marker='x')
                        ax0.legend()
                        ax1.plot(bin_mag, target_moment_per_bin, label='target_moment_per_bin', marker='x')
                        ax1.plot(bin_mag, moment_rate_in_bin, label='moment_rate_in_bin', marker='x')
                        ax1.set_yscale('log')
                        ax1.legend()
                        plt.savefig(self.pathlog + '/Log_tmp_' + str(self.sample) + '.png', dpi=80, transparent=True)
                        plt.close()
                'Pick the bin of magnitude to fill according to the current distribution '
                target_i = target_i / sum(target_i)
                picked_bin = np.random.choice(len(bin_mag), 1, p=target_i)[0]
                mag = bin_mag[picked_bin]
                tmp = time.time()
                if not picked_bin in empty_bins:
                    empty_in_bin = set(rup_in_bin[picked_bin]) & set(empty_rups)
                    if len(empty_in_bin) == len(rup_in_bin[picked_bin]):
                        empty_bins.append(picked_bin)
                time_checking_empty_bin += time.time() - tmp
                if not picked_bin in empty_bins:
                    tmp = time.time()
                    'Calculate the weight for sampling of the fault or scenario'
                    do_rup_weight = False
                    if number_of_loops == 1:
                        do_rup_weight = True
                    if sum_fault_budget < weigthing_built[weigth_rup_sample]:
                        do_rup_weight = True
                    if do_rup_weight == True:
                        loop_last_rup_w = number_of_loops
                        weigth_rup_sample += 1
                        if faster_rup_weight == True:
                            list_of_bins = range(len(bin_mag))
                        else:
                            list_of_bins = [picked_bin]
                        w_rup_binmag = []
                        for index_mag in list_of_bins:
                            if index_mag in empty_bins:
                                w_rup_binmag.append([])
                            else:
                                weight_rup_i = core_utils.weight_fault_sampling(index_mag, rup_in_bin, faults_names, faults_slip_rates, slip_rate_use_per_fault, faults_alone, scenarios_names, faults_isolated, index_faults_in_scenario, rup_rates, empty_rups)
                                if local_MFD == True:
                                    factor_on_weight = core_utils.check_local_mfd(rup_rates, rup_in_bin[index_mag], index_mag, bin_mag, local_mfds, associated_rup, associated_weight)
                                    weight_rup_i = np.array([i * w for (i, w) in zip(weight_rup_i, factor_on_weight)])
                                    weight_rup_i /= weight_rup_i.sum()
                                if sum(weight_rup_i) != 0.0:
                                    weight_rup_i = [float(i) / sum(weight_rup_i) for i in weight_rup_i]
                                w_rup_binmag.append(weight_rup_i)
                        time_weight_rupt += time.time() - tmp
                        tmp = time.time()
                        for index_mag in list_of_bins:
                            if not index_mag in empty_bins:
                                weight_rup = w_rup_binmag[index_mag]
                                if set(rup_in_bin[index_mag]) & set(empty_rups):
                                    i = 0
                                    for i_rup in rup_in_bin[index_mag]:
                                        if str(i_rup) in empty_rups:
                                            weight_rup[i] = 0.0
                                        i += 1
                                weight_rup = list(weight_rup)
                                sum_weight_rup = sum(weight_rup)
                                if sum(weight_rup) == 0.0:
                                    empty_bins.append(index_mag)
                                if not index_mag in empty_bins:
                                    if sum_weight_rup != 1.0:
                                        weight_rup = [float(i) / sum_weight_rup for i in weight_rup]
                                    if math.isnan(sum(weight_rup)):
                                        print('WARNING : sum rup weight is nan')
                                        nb_nans = 0
                                        id = 0
                                        for i in weight_rup:
                                            if math.isnan(i):
                                                nb_nans += 1
                                            weight_rup[id] = 0.0
                                            id += 1
                                w_rup_binmag[index_mag] = weight_rup
                        time_clean_w_rupt += time.time() - tmp
                    if faster_rup_weight == True:
                        weight_rup = w_rup_binmag[picked_bin]
                    else:
                        weight_rup = w_rup_binmag[0]
                    try:
                        i_picked = np.where(np.random.multinomial(1, weight_rup) == 1)[0][0]
                        picked_rup = rup_in_bin[picked_bin][i_picked]
                    except ValueError:
                        print("rupt weights didn't work. sum:", sum(weight_rup))
                        picked_rup = np.random.choice(rup_in_bin[picked_bin])
                        n_w_crash += 1
                    index_fault = rup_rates.get(str(picked_rup)).get('involved_faults')
                    if bool_target_set == False:
                        tmp = time.time()
                        last_bins_empty = True
                        for bin_i in range(len(bin_mag))[-3:]:
                            if not bin_i in empty_bins:
                                last_bins_empty = False
                        if last_bins_empty == True:
                            rate_tot_model = rates.get_rate_model(rup_rates, fault_prop, bin_mag)
                            bool_target_set = True
                            print('set target - limit on the largest magnitudes')
                        if moment_rate_left <= (1.0 - 1e-05) * moment_rate_required:
                            self.calculation_log_file.write('\n Not enough moment left ')
                            for rup_i in rup_in_bin[-3] + rup_in_bin[-2] + rup_in_bin[-1]:
                                if not str(rup_i) in empty_rups:
                                    empty_rups.append(str(rup_i))
                            rup_in_bin[-3] = []
                            rup_in_bin[-2] = []
                            rup_in_bin[-1] = []
                            bool_target_set = True
                            print('set target - moment rate is limiting')
                        antelast_bins_empty = True
                        for bin_i in range(len(bin_mag))[-2:]:
                            if not bin_i in empty_bins:
                                antelast_bins_empty = False
                        if antelast_bins_empty == True and bool_target_set == False:
                            if moment_rate_in_bin[-3] >= 2.0 * (moment_rate_in_bin[-2] + moment_rate_in_bin[-1]):
                                self.calculation_log_file.write('\n antepenultimate bin getting too high')
                                for rup_i in rup_in_bin[-3]:
                                    if not str(rup_i) in empty_rups:
                                        empty_rups.append(str(rup_i))
                                empty_bins.append(range(len(bin_mag))[-3])
                                bool_target_set = True
                                print('set target - antepenultimate bin getting too high')
                        if number_of_loops > number_of_loops_last_checked + 50.0:
                            number_of_loops_last_checked = number_of_loops
                            moment_rate_left = Total_moment_faults_rate_init - Total_moment_rate_fault
                            rate_tot_model = rates.get_rate_model(rup_rates, fault_prop, bin_mag)
                            rate_Mmax_check = np.mean(rate_tot_model[-3:])
                            moment_rate_required = 0.0
                            for index_mag in range(len(bin_mag) - 3):
                                rate_Mi_check = rate_tot_model[index_mag]
                                target_GR_i_check = rate_Mmax_check * p_MFD[index_mag] / p_MFD[-2]
                                moment_rate_required += (mag_to_M0(bin_mag[index_mag]) * target_GR_i_check - mag_to_M0(bin_mag[index_mag]) * rate_Mi_check) * fault_prop(bin_mag[picked_bin])
                            if self.mfd_hyp == 'UCERF_DV_':
                                rate_Mmax = rate_tot_model[-1]
                                moment_rate_required = 0.0
                                for index_mag in range(len(bin_mag) - 3):
                                    rate_Mi_check = rate_tot_model[index_mag]
                                    target_GR_i_check = rate_Mmax_check * p_MFD[index_mag] / p_MFD[-3]
                                    moment_rate_required += (mag_to_M0(bin_mag[index_mag]) * target_GR_i_check - mag_to_M0(bin_mag[index_mag]) * rate_Mi_check) * fault_prop(bin_mag[picked_bin])
                        time_checking_target_reach += time.time() - tmp
                        if do_the_target == True and bool_target_set == True:
                            rate_tot_model = rates.get_rate_model(rup_rates, fault_prop, bin_mag)
                            do_the_target = False
                            print('- target set - ')
                            self.calculation_log_file.write('\n- target set - ')
                            rate_at_target_setting = rate_tot_model
                            rate_Mmax = np.mean(rate_tot_model[-3:])
                            TARGET = []
                            for t_mag_bin in range(len(bin_mag)):
                                TARGET.append(rate_Mmax * p_MFD[t_mag_bin] / p_MFD[-2])
                            if self.mfd_hyp == 'UCERF_DV_':
                                rate_Mmax = rate_tot_model[-1]
                                TARGET = []
                                for t_mag_bin in range(len(bin_mag)):
                                    rate_Mi = rate_tot_model[t_mag_bin]
                                    TARGET.append(rate_Mmax * p_MFD[picked_bin] / p_MFD[-3])
                    if picked_bin in bin_target_reached:
                        print('WHAT ARE YOU DOING HERE?', bin_mag[picked_bin])
                        exit()
                    ' spending the slip_rate increment '
                    tmp = time.time()
                    index_fault = rup_rates.get(str(picked_rup)).get('involved_faults')
                    sr_to_spend = True
                    shear_mod = 0
                    for index in index_fault:
                        shear_mod += faults_shear_mod[index]
                        if faults_budget[index] <= 0.0:
                            sr_to_spend = False
                    if sr_to_spend == False:
                        if not str(picked_rup) in empty_rups:
                            empty_rups.append(str(picked_rup))
                        else:
                            picked_empty_rup += 1
                    if sr_to_spend == True:
                        shear_mod = shear_mod / float(len(index_fault))
                        area = rup_rates.get(str(picked_rup)).get('area')
                        displacement = mag_to_M0(mag) / (shear_mod * area)
                        rate_i = size_of_increment / displacement
                        if option_fast == True:
                            min_budget_local = min([faults_budget[i] for i in index_fault])
                            nb_loop_spending = int(min_budget_local / min_budget)
                            if nb_loop_spending < 1:
                                nb_loop_spending = 1
                        else:
                            nb_loop_spending = 1
                        if bool_target_set == True:
                            rate_Mi = rate_in_model[picked_bin]
                            target_mfd_i = TARGET[picked_bin] * fault_prop(bin_mag[picked_bin])
                            if rate_Mi < target_mfd_i:
                                if uniform_spending == True or len(index_fault) == 1:
                                    moment_rate_i = 0.0
                                    for index in index_fault:
                                        for loop_spending in range(nb_loop_spending):
                                            M_slip_repartition[str(faults_names[index])][str(picked_rup)] += 1
                                        faults_budget[index] += -1 * nb_loop_spending
                                        sum_fault_budget += -1 * nb_loop_spending
                                        slip_rate_use_per_fault[index] += size_of_increment * nb_loop_spending
                                    rup_rates[str(picked_rup)]['rates'][picked_bin] += rate_i * nb_loop_spending
                                    rate_in_model[picked_bin] += rate_i * nb_loop_spending
                                    moment_rate_i += mag_to_M0(mag) * rate_i * nb_loop_spending
                                else:
                                    moment_rate_i = 0.0
                                    for loop_spending in range(nb_loop_spending):
                                        (M_slip_repartition, faults_budget, slip_rate_use_per_fault, nb_sdr_used, sum_fault_budget) = core_utils.variable_spending(index_fault, M_slip_repartition, faults_budget, slip_rate_use_per_fault, size_of_increment, faults_slip_rates, picked_rup, faults_names, sum_fault_budget)
                                        rup_rates[str(picked_rup)]['rates'][picked_bin] += rate_i * nb_sdr_used
                                        rate_in_model[picked_bin] += rate_i * nb_sdr_used
                                        moment_rate_i += mag_to_M0(mag) * rate_i * nb_sdr_used
                                moment_rate_in_bin[picked_bin] += moment_rate_i
                                Total_moment_rate_fault += moment_rate_i
                            else:
                                rup_in_bin[picked_bin] = []
                                if not picked_bin in bin_target_reached:
                                    bin_target_reached.append(picked_bin)
                        else:
                            if uniform_spending == True or len(index_fault) == 1:
                                moment_rate_i = 0.0
                                for index in index_fault:
                                    for loop_spending in range(nb_loop_spending):
                                        M_slip_repartition[str(faults_names[index])][str(picked_rup)] += 1
                                    faults_budget[index] += -1 * nb_loop_spending
                                    sum_fault_budget += -1 * nb_loop_spending
                                    slip_rate_use_per_fault[index] += size_of_increment * nb_loop_spending
                                rup_rates[str(picked_rup)]['rates'][picked_bin] += rate_i * nb_loop_spending
                                rate_in_model[picked_bin] += rate_i * nb_loop_spending
                                moment_rate_i += mag_to_M0(mag) * rate_i * nb_loop_spending
                            else:
                                moment_rate_i = 0.0
                                for loop_spending in range(nb_loop_spending):
                                    (M_slip_repartition, faults_budget, slip_rate_use_per_fault, nb_sdr_used, sum_fault_budget) = core_utils.variable_spending(index_fault, M_slip_repartition, faults_budget, slip_rate_use_per_fault, size_of_increment, faults_slip_rates, picked_rup, faults_names, sum_fault_budget)
                                    rup_rates[str(picked_rup)]['rates'][picked_bin] += rate_i * nb_sdr_used
                                    rate_in_model[picked_bin] += rate_i * nb_sdr_used
                                    moment_rate_i += mag_to_M0(mag) * rate_i * nb_sdr_used
                            moment_rate_in_bin[picked_bin] += moment_rate_i
                            Total_moment_rate_fault += moment_rate_i
                    time_spending_dsr += time.time() - tmp
                elif not picked_bin in empty_bins:
                    empty_bins.append(picked_bin)
                tmp = time.time()
                if number_of_loops > number_of_loops_before + 500:
                    number_of_loops_before = number_of_loops
                    for index_mag in range(len(bin_mag)):
                        if not index_mag in empty_bins:
                            nb_rup_not_empty = 0
                            for i_rup in rup_in_bin[index_mag]:
                                if nb_rup_not_empty == 0:
                                    rup_is_empty = False
                                    for index_fault in rup_rates.get(str(i_rup)).get('involved_faults'):
                                        if faults_budget[index_fault] <= 0:
                                            rup_is_empty = True
                                    if rup_is_empty == False:
                                        nb_rup_not_empty += 1
                            if nb_rup_not_empty == 0:
                                empty_bins.append(index_mag)
                time_checking_empty_bin += time.time() - tmp
                tmp = time.time()
                len_faults_budget.append(sum_fault_budget)
                if len(len_faults_budget) > 3:
                    if len_faults_budget[-2] == len_faults_budget[-1]:
                        number_of_loops_for_nothing += 1
                    if number_of_loops_for_nothing_before < number_of_loops_for_nothing - 100:
                        number_of_loops_for_nothing_before = number_of_loops_for_nothing
                        if len_faults_budget[-1] == len_faults_budget[-10]:
                            rup_still_used = [i for i in rup_in_bin[picked_bin] if not i in empty_rups]
                            fault_still_used = []
                            for rup_i in rup_still_used:
                                fault_still_used += list(rup_rates.get(str(rup_i)).get('involved_faults'))
                            fault_still_used = set(fault_still_used)
                            for (fault, index_fault) in zip(faults_names, range(len(faults_names))):
                                if not index_fault in fault_still_used and faults_budget[index_fault] > 0:
                                    while faults_budget[index_fault] > 0:
                                        ratio_done = 1.0 - float(sum_fault_budget) / nb_ss_to_spend
                                        if ratio_done > 0.01:
                                            (model_MFD, self.calculation_log_file, print_percent) = core_utils.progress(model_MFD, self.calculation_log_file, ratio_done, print_percent, rup_rates, fault_prop, bin_mag)
                                        faults_budget[index_fault] += -1
                                        sum_fault_budget += -1
                                        M_slip_repartition[str(faults_names[index_fault])]['NMS'] += 1
                                        aseismic_count += 1
                time_checking_empty_faults += time.time() - tmp
            else:
                print('-target filled-')
                self.calculation_log_file.write('\n-target filled-')
                while sum_fault_budget != 0:
                    ratio_done = 1.0 - float(sum_fault_budget) / nb_ss_to_spend
                    if ratio_done > 0.01:
                        (model_MFD, self.calculation_log_file, print_percent) = core_utils.progress(model_MFD, self.calculation_log_file, ratio_done, print_percent, rup_rates, fault_prop, bin_mag)
                    for index_fault in range(len(faults_names)):
                        if faults_budget[index_fault] > 0:
                            faults_budget[index_fault] += -1
                            sum_fault_budget += -1
                            M_slip_repartition[str(faults_names[index_fault])]['NMS'] += 1
                            aseismic_count += 1
        ' check if the TARGET as been set.\n        if not, build it for comparing'
        if TARGET == []:
            rate_tot_model = rates.get_rate_model(rup_rates, fault_prop, bin_mag)
            print('- target set at the end- ')
            self.calculation_log_file.write('\n- target set at the end - ')
            rate_at_target_setting = rate_tot_model
            rate_Mmax = np.mean(rate_tot_model[-3:])
            for t_mag_bin in range(len(bin_mag)):
                TARGET.append(rate_Mmax * p_MFD[t_mag_bin] / p_MFD[-2])
        '##################################################################\n        #\n        # Definition of the background seismicity\n        #\n        ##################################################################'
        (rate_f_in_model, rate_bg_in_model) = rates.get_rate_faults_n_bg(rup_rates, fault_prop, bin_mag)
        EQ_rate_BG = rate_bg_in_model
        '##################################################################\n        # printing\n        ##################################################################'
        rate_tot_model = rates.get_rate_model(rup_rates, fault_prop, bin_mag)
        model_MFD.append(rate_tot_model)
        colors = ['gainsboro', 'darkgray', 'dimgray', 'dimgray', 'black', 'red']
        index_color = 0
        for MFD_i in model_MFD:
            plt.plot(bin_mag, MFD_i, color=colors[index_color])
            index_color += 1
        if do_the_target == True:
            rate_at_target_setting = model_MFD[-1]
        plt.plot(bin_mag, rate_at_target_setting, ':g')
        plt.plot(bin_mag, TARGET, ':b')
        plt.scatter(bin_mag[-3:], rate_tot_model[-3:], c='k')
        plt.yscale('log')
        plt.savefig(self.pathlog + '/Log_target_fit_' + str(self.sample) + '.png', dpi=180, transparent=True)
        plt.close()
        index_5 = 0
        while bin_mag[index_mag] < 5.0:
            index_5 += 1
        self.ratio_test = np.mean([abs(TARGET[index_5 + 0] / rate_tot_model[index_5 + 0]), abs(TARGET[index_5 + 1] / rate_tot_model[index_5 + 1]), abs(TARGET[index_5 + 2] / rate_tot_model[index_5 + 2]), abs(TARGET[index_5 + 3] / rate_tot_model[index_5 + 3]), abs(TARGET[index_5 + 4] / rate_tot_model[index_5 + 4])])
        MFD_to_test = rate_tot_model / sum(rate_tot_model)
        target_to_test = TARGET / sum(TARGET)
        array_absolute_ratio = []
        for index_mag in range(len(bin_mag)):
            array_absolute_ratio.append(abs(MFD_to_test[index_mag] / target_to_test[index_mag]))
        self.ratio_test = np.mean(array_absolute_ratio)
        print('ratio between the target and the shape of the model : ' + str(round(self.ratio_test, 2)))
        self.calculation_log_file.write('\nratio between the target and the shape of the model : ' + str(round(self.ratio_test, 2)))
        '##################################################################\n        #\n        # OUTPUT for Source_model_creator\n        #\n        ##################################################################'
        self.OQ_entry_faults = []
        self.OQ_entry_scenarios = []
        for i in range(len(faults_names)):
            self.OQ_entry_faults.append(rup_rates.get(str(i)).get('rates'))
        i_end = i + 1
        if np.size(scenarios_names) != 0:
            for j in range(len(scenarios_names)):
                self.OQ_entry_scenarios.append(rup_rates.get(str(i_end + j)).get('rates'))
        self.faults_names = faults_names
        self.scenarios_names = scenarios_names
        self.EQ_rate_BG = EQ_rate_BG
        self.bin_mag = bin_mag
        self.rup_rates = rup_rates
        self.M_slip_repartition = M_slip_repartition
        log_calculation_file.write('Moment_rate_faults_final calculated using 10^(1.5M+9.1)' + '\t' + str(Total_moment_rate_fault) + '\t' + 'N.m' + '\n')
        log_calculation_file.write('Ratio of NMS slip rate counted in the iterative process: ' + '\t' + str(round(100.0 * (aseismic_count / nb_ss_to_spend))) + '\t %\n')
        log_calculation_file.write('Moment_rate_model calculated using 10^(1.5M+9.1)' + '\t' + str(Total_moment_rate_fault) + '\t' + 'N.m' + '\n')
        log_calculation_file.close()
        self.ratio_NMS = round(100.0 * (1.0 - Total_moment_rate_fault / Total_moment_faults_rate_init))
        print('ratio of NMS : ' + str(round(100.0 * (aseismic_count / nb_ss_to_spend))))
        self.calculation_log_file.write('\nratio of NMS : ' + str(round(100.0 * (aseismic_count / nb_ss_to_spend))))
        f = open(log_sliprep_file, 'wb')
        pickle.dump(M_slip_repartition, f)
        f.close()