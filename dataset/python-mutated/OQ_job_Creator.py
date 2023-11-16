"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.0 

This code is the main code for building OQ entries.
It creates the job.ini and launch the other codes creating logic trees (Sources and GMPEs) and sources models.

@author: Thomas Chartier
"""
import os
import numpy as np
''
'        Reading input data          '
''
' input data are located in the run.info and job.ini file'

class OQ_job_Creator:

    def __init__(self, Run_Name):
        if False:
            while True:
                i = 10
        self.Run_Name = Run_Name
        self.input_data()
        self.write_ini()

    def input_data(self):
        if False:
            while True:
                i = 10
        info_run_file = 'input/' + str(self.Run_Name) + '/run.info'
        if os.path.exists(info_run_file):
            read_info_file = open(info_run_file, 'r')
            lines_of_the_file = read_info_file.readlines()
            self.option_map = lines_of_the_file[1].split('\t')[1]
            lines_of_the_file = [x.strip('L\n') for x in lines_of_the_file]
            lines_of_the_file = [x.strip('\r\n') for x in lines_of_the_file]
            lines_of_the_file = [x.strip('\n') for x in lines_of_the_file]
            if self.option_map == 'yes':
                self.ll_lon = lines_of_the_file[2].split('\t')[1]
                self.ll_lat = lines_of_the_file[2].split('\t')[2]
                self.lr_lon = lines_of_the_file[2].split('\t')[3]
                self.lr_lat = lines_of_the_file[2].split('\t')[4]
                self.ur_lon = lines_of_the_file[2].split('\t')[5]
                self.ur_lat = lines_of_the_file[2].split('\t')[6]
                self.ul_lon = lines_of_the_file[2].split('\t')[7]
                self.ul_lat = lines_of_the_file[2].split('\t')[8]
                self.region_grid_spacing = lines_of_the_file[3].split('\t')[1]
            else:
                self.Longitude = lines_of_the_file[2].split('\t')[1]
                self.Latitude = lines_of_the_file[3].split('\t')[1]
            self.Vs30 = lines_of_the_file[4].split('\t')[1]
            self.Z1000 = lines_of_the_file[5].split('\t')[1]
            self.Z2500 = lines_of_the_file[6].split('\t')[1]
            self.nb_LT_samp = lines_of_the_file[7].split('\t')[1]
            self.rup_mesh = lines_of_the_file[8].split('\t')[1]
            self.source_discr = lines_of_the_file[9].split('\t')[1]
            self.investigation_time = lines_of_the_file[10].split('\t')[1]
            self.poes = lines_of_the_file[11].split('\t')[1]
            self.trunc_lvl = lines_of_the_file[12].split('\t')[1]
            self.max_dist = lines_of_the_file[13].split('\t')[1]
            self.nb_sample = int(lines_of_the_file[14].split('\t')[1])
            self.Mmin = float(lines_of_the_file[15].split('\t')[1])
            self.seed = int(lines_of_the_file[16].split('\t')[1])
            self.sr_correl = lines_of_the_file[17].split('\t')[1]
            if self.sr_correl == 'yes' or self.sr_correl == 'Yes' or self.sr_correl == 'True' or (self.sr_correl == 'true'):
                self.sr_correl = True
            else:
                self.sr_correl = False
            self.size_of_increment = float(lines_of_the_file[18].split('\t')[1])
            self.Mmax_range = [float(lines_of_the_file[19].split('\t')[1]), float(lines_of_the_file[19].split('\t')[2])]
            self.intensity_out = []
            self.min_int_out = []
            self.max_int_out = []
            self.nb_int_out = []
            for line in lines_of_the_file[20:]:
                self.intensity_out.append(line.split('\t')[1])
                self.min_int_out.append(float(line.split('\t')[2]))
                self.max_int_out.append(float(line.split('\t')[3]))
                self.nb_int_out.append(int(line.split('\t')[4]))
            read_info_file.close()
        else:
            print('ERROR : File run.info not found! \n Please refer to the user manual')
            exit()

    def write_ini(self):
        if False:
            i = 10
            return i + 15
        jobfile = open(str(self.Run_Name) + '/job.ini', 'w')
        Ligne = '[general]\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        jobfile.write(Ligne)
        Ligne = "description = '" + str(self.Run_Name) + "'\n"
        jobfile.write(Ligne)
        Ligne = 'calculation_mode = classical\n'
        jobfile.write(Ligne)
        Ligne = 'random_seed = ' + str(self.seed) + '\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        jobfile.write(Ligne)
        Ligne = '[geometry]\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        if self.option_map == 'yes':
            jobfile.write(Ligne)
            Ligne = 'region = ' + str(self.ll_lon) + ' ' + str(self.ll_lat) + ', ' + str(self.lr_lon) + ' ' + str(self.lr_lat) + ', ' + str(self.ur_lon) + ' ' + str(self.ur_lat) + ', ' + str(self.ul_lon) + ' ' + str(self.ul_lat) + '\n'
            jobfile.write(Ligne)
            Ligne = 'region_grid_spacing = ' + str(self.region_grid_spacing) + '\n'
        else:
            jobfile.write(Ligne)
            Ligne = 'sites = ' + str(self.Longitude)[:-1] + ' ' + str(self.Latitude) + '\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        jobfile.write(Ligne)
        Ligne = '[logic_tree]\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        jobfile.write(Ligne)
        Ligne = 'number_of_logic_tree_samples = ' + str(self.nb_LT_samp) + '\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        jobfile.write(Ligne)
        Ligne = '[erf]\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        jobfile.write(Ligne)
        Ligne = 'rupture_mesh_spacing =' + str(self.rup_mesh) + '\n'
        jobfile.write(Ligne)
        Ligne = 'width_of_mfd_bin = 0.1\n'
        jobfile.write(Ligne)
        Ligne = 'area_source_discretization = ' + str(self.source_discr) + '\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        jobfile.write(Ligne)
        Ligne = '[site_params]\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        jobfile.write(Ligne)
        Ligne = 'reference_vs30_type = measured\n'
        jobfile.write(Ligne)
        Ligne = 'reference_vs30_value = ' + str(self.Vs30) + '\n'
        jobfile.write(Ligne)
        Ligne = 'reference_depth_to_2pt5km_per_sec = ' + str(self.Z2500) + '\n'
        jobfile.write(Ligne)
        Ligne = 'reference_depth_to_1pt0km_per_sec = ' + str(self.Z1000) + '\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        jobfile.write(Ligne)
        Ligne = '[calculation]\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        jobfile.write(Ligne)
        Ligne = 'source_model_logic_tree_file = Sources_Logic_tree.xml\n'
        jobfile.write(Ligne)
        Ligne = 'gsim_logic_tree_file = GMPE_Logic_tree.xml\n'
        jobfile.write(Ligne)
        Ligne = 'investigation_time = ' + str(self.investigation_time) + '\n'
        jobfile.write(Ligne)
        Ligne = 'intensity_measure_types_and_levels = {'
        jobfile.write(Ligne)
        intensities_done = []
        for i in range(len(self.intensity_out)):
            Ligne_1 = '"' + str(self.intensity_out[i]) + '": ['
            array_ints = np.logspace(np.log10(self.min_int_out[i]), np.log10(self.max_int_out[i]), self.nb_int_out[i])
            Ligne_2 = ''
            for values in array_ints:
                if len(Ligne_2) > 1:
                    Ligne_2 += ', '
                Ligne_2 += str(values)
            if not self.intensity_out[i] in intensities_done:
                if len(intensities_done) != 0:
                    jobfile.write(', ')
                jobfile.write(Ligne_1 + Ligne_2 + ']')
            intensities_done.append(self.intensity_out[i])
        Ligne = '}\n'
        jobfile.write(Ligne)
        Ligne = 'truncation_level = ' + str(self.trunc_lvl) + '\n'
        jobfile.write(Ligne)
        Ligne = 'maximum_distance = ' + str(self.max_dist) + '\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        jobfile.write(Ligne)
        Ligne = '[output]\n'
        jobfile.write(Ligne)
        Ligne = '\n'
        jobfile.write(Ligne)
        Ligne = 'export_dir =  ./results\n'
        jobfile.write(Ligne)
        Ligne = 'mean_hazard_curves = true\n'
        jobfile.write(Ligne)
        Ligne = 'quantile_hazard_curves =\n'
        jobfile.write(Ligne)
        Ligne = 'uniform_hazard_spectra = false\n'
        jobfile.write(Ligne)
        Ligne = 'poes = ' + str(self.poes) + '\n'
        jobfile.write(Ligne)