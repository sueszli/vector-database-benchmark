"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.0

This code create a source logic tree

@author: Thomas Chartier
"""
import xml.etree.ElementTree as ET
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import sys
from shutil import copyfile
path_actuel = os.path.dirname(os.path.abspath(__file__))
path_lib = path_actuel + '/lib'
sys.path.append(path_lib)
path_f = path_lib + '/file_writer'
sys.path.append(path_f)
path_f = path_lib + '/utils'
sys.path.append(path_f)
from Source_Model import *
import BG_ratio
import faults_n_scenarios
from OQ_job_Creator import OQ_job_Creator
import read_input
import geojson
import pickle
from geometry_tools import distance
import toml
import glob

class Sources_Logic_Tree_Creator:

    def __init__(self, param, calculation_log_file):
        if False:
            for i in range(10):
                print('nop')
        self.param = param
        path = self.param['dirpath']
        self.Run_Name = param['Run_Name']
        if param['main']['fault_input_type'] == 'geojson':
            self.faults_file = path + param['main']['faults_file']
            self.File_geom = self.faults_file
        elif param['main']['fault_input_type'] == 'txtsherifs':
            self.File_geom = path + param['main']['File_geom']
            self.File_prop = path + param['main']['File_prop']
        if param['main']['background']['option_bg'] == 'smooth':
            fbgpath = param['dirpath'] + param['main']['background']['smoothing_xml']
            if os.path.isdir(fbgpath):
                list_fbg = [f for f in listdir(fbgpath) if isfile(join(fbgpath, f))]
            else:
                list_fbg = fbgpath.split(' ')
                while '' in list_fbg:
                    list_fbg.remove('')
        else:
            list_fbg = []
            fbgpath = None
        self.list_fbg = list_fbg
        self.fbgpath = fbgpath
        self.nb_random_sampling = param['main']['parameters']['nb_sample']
        self.overwrite = param['main']['parameters']['overwrite_files']
        self.calculation_log_file = calculation_log_file
        self.Domain_in_model = []
        self.initialize()

    def initialize(self):
        if False:
            return 10
        path = self.param['dirpath']
        LT_file = path + self.Run_Name + '/Sources_Logic_tree.xml'
        LT_log_name = path + self.param['main']['LT_file']
        lt_info_file = open(path + self.Run_Name + '/ssm/lt_branches_id.txt', 'w')
        lt_info_file.write('id\tmodel\tmfd\trup_set\tbackground\tscaling\t')
        lt_info_file.write('sample\n')
        if not os.path.exists(LT_log_name):
            print('ERROR : Please provide a LT_file file \n              See the user manual for guidelines and               the example for file setup example.')
            exit()
        else:
            LT = toml.load(LT_log_name)
            model_hyps = LT['Models']
            mfd_hyps = LT['MFD_shape']
            bg_hyps = LT['Background']
            sc_hyps = LT['scenario_set']
            scL_hyps = LT['Scaling_Laws']
            branches = []
            for model in model_hyps:
                for mfd in mfd_hyps:
                    for bg in bg_hyps:
                        for sc in sc_hyps:
                            for scl in scL_hyps:
                                bi = [model, mfd, bg, sc, scl]
                                branches.append(bi)
            tmp = branches
            branches = []
            b_indexes = []
            for i_b in range(len(tmp)):
                for sample in range(1, self.nb_random_sampling + 1):
                    bi = tmp[i_b] + [sample]
                    branches.append(bi)
                    b_indexes.append(i_b)
            force_rerun = self.param['main']['parameters']['force_rerun']
            if force_rerun in ['False', 'false']:
                if os.path.isfile(path + self.Run_Name + '/LOG/lt_branchs.pkl'):
                    old_branches = pickle.load(open(path + self.Run_Name + '/LOG/lt_branchs.pkl', 'rb'))
                    old_indexes = pickle.load(open(path + self.Run_Name + '/LOG/lt_b_id.pkl', 'rb'))
                else:
                    old_branches = []
                    old_indexes = []
            else:
                old_branches = []
                old_indexes = []
            dict_LT = {}
            used_id = []
            for bi in branches:
                if force_rerun in ['False', 'false']:
                    if bi in old_branches:
                        rerun_bi = False
                        i = old_branches.index(bi)
                        id = old_indexes[i]
                        if not id in used_id:
                            used_id.append(id)
                    else:
                        rerun_bi = True
                        id = None
                else:
                    rerun_bi = True
                    id = None
                if rerun_bi == True:
                    i = 0
                    while i in used_id + old_indexes:
                        i += 1
                    id = i
                if not id in used_id:
                    used_id.append(id)
                if self.overwrite in ['True', 'true']:
                    rerun_bi = True
                dict_LT.update({id: {'run_branch': rerun_bi, 'model': bi[0], 'mfd': bi[1], 'set': bi[3], 'bg': bi[2], 'scl': bi[4], 'smp': bi[5]}})
                lt_info_file.write(str(id) + '\t')
                lt_info_file.write(str(bi[0]) + '\t')
                lt_info_file.write(' '.join([i for i in bi[1]]) + '\t')
                lt_info_file.write(str(bi[3]) + '\t')
                lt_info_file.write(str(bi[2]) + '\t')
                lt_info_file.write(' '.join([i for i in bi[4]]) + '\t')
                lt_info_file.write(str(bi[5]) + '\n')
            lt_info_file.close()
            if not self.param['main']['background']['option_bg'] in ['None', 'none']:
                try:
                    available_bg = read_input.extract_bg_input(path + 'input/' + self.Run_Name + '/bg_seismicity.txt')
                except:
                    print('Error related to the background file \n' + 'Please make sure input/run_name/bg_seismicity.txt                      is correctly set up')
            else:
                print('No background is used')
            try:
                rupt_file = path + self.param['main']['rupture_file']
                available_sets = read_input.extract_sc_input(rupt_file)
            except:
                print('Error related to the rupture scenario set file \n' + 'Please make sure input/run_name/ruptures.txt is correctly set up')
        last_bg = 'impossible_name'
        last_set = 'impossible_name'
        last_model = 'impossible_name'
        line = "<?xml version='1.0' encoding='utf-8'?>\n"
        line += '<nrml xmlns:gml="http://www.opengis.net/gml"\n'
        line += '\txmlns="http://openquake.org/xmlns/nrml/0.5">\n'
        line += '\t<logicTree logicTreeID="lt1">\n'
        line += '\t\t<logicTreeBranchingLevel branchingLevelID="bl_1">\n'
        line += '\t\t\t<logicTreeBranchSet uncertaintyType="sourceModel"\n'
        line += '\t\t\t\t\t\t\tbranchSetID="bs_1">\n'
        for id in used_id:
            print('\n*******\nLOGIC TREE BRANCH', id, '\n*******')
            branch = dict_LT[id]
            model_hyp = branch['model']
            scl_hyp = branch['scl']
            mfd_hyp = branch['mfd']
            bg_hyp = branch['bg']
            set_hyp = branch['set']
            smp = branch['smp']
            if branch['run_branch'] == True:
                b_path = path + self.Run_Name + '/ssm/b_' + str(id)
                log_path = path + self.Run_Name + '/ssm/log_b_' + str(id)
                if not os.path.exists(b_path):
                    os.makedirs(b_path)
                    print('running branch id ', str(id), ' for the first time')
                elif self.param['main']['parameters']['force_rerun'] in ['true', 'True']:
                    files = glob.glob(b_path + '/*')
                    for f in files:
                        os.remove(f)
                    print('rerunning branch id ', str(id))
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                    '\n                    files = glob.glob(log_path+"/participation_rates/*")\n                    for f in files:\n                        os.remove(f)\n                    os.dir\n                    files = glob.glob(log_path+"/*")\n                    for f in files:\n                        os.remove(f)\n                    '
                line += '\t\t\t\t<logicTreeBranch branchID= "b_' + str(id) + '">\n'
                self.calculation_log_file.write('\n\nRunning logic tree branch:')
                print()
                print('Model : \t', model_hyp)
                print('Rupture set : \t\t', set_hyp)
                if mfd_hyp[0] in ['GR', 'YC', 'tapered_GR']:
                    print('Model : \t\t\t', mfd_hyp[0], ' b : ', mfd_hyp[1])
                elif mfd_hyp[0] == 'YC_modified':
                    print('Model : \t\t\t', mfd_hyp[0], ' b : ', mfd_hyp[1], ' Mf : ', mfd_hyp[2], ' size_of_bump : ', mfd_hyp[3])
                elif mfd_hyp[0] == 'double_GR':
                    print('Model : \t\t\t', mfd_hyp[0], ' b : ', mfd_hyp[1], ' Mrupt : ', mfd_hyp[2])
                print('Model : \t\t\t\t', bg_hyp)
                print('Scaling law : \t\t\t\t\t', ' '.join((i for i in scl_hyp)))
                print('Sample : ', smp)
                print()
                self.calculation_log_file.write('\n' + str(model_hyp) + '-' + str(scl_hyp) + '-' + str(mfd_hyp) + '-' + str(bg_hyp) + '-' + str(set_hyp) + '-' + str(set_hyp) + '-' + str(smp))
                if last_bg != bg_hyp:
                    bg_ratio = available_bg[bg_hyp]
                    last_bg = bg_hyp
                if last_set != set_hyp:
                    rupture_set = available_sets[set_hyp]
                    index_scenario = 0
                    scenarios_names = []
                    if np.size(rupture_set) == 0:
                        scenarios_names = []
                    else:
                        for index_scenario in range(len(rupture_set)):
                            faults_in_scenario = rupture_set[index_scenario]
                            if len(faults_in_scenario) > 1:
                                scenario = {}
                                faults_done = []
                                for i in range(len(faults_in_scenario)):
                                    if not str(faults_in_scenario[i]).replace('\r', '') in faults_done:
                                        scenario['f_%s' % str(i + 1)] = [str(faults_in_scenario[i]).replace('\r', '').replace('\t', '').replace('\n', '')]
                                        faults_done.append(str(faults_in_scenario[i]).replace('\r', '').replace('\t', '').replace('\n', ''))
                                if len(scenario) != 0:
                                    scenarios_names.append(scenario)
                            index_scenario += 1
                    last_set = set_hyp
                if last_model != model_hyp:
                    last_model = model_hyp
                    print('Importing faults')
                    if self.param['main']['fault_input_type'] == 'txtsherifs':
                        Prop = np.genfromtxt(self.File_prop, dtype=['U100', 'U100', 'f8', 'U100', 'U100', 'f8', 'f8', 'f8', 'f8', 'f8', 'U100', 'f8'], skip_header=1)
                        Column_model_name = list(map(lambda i: Prop[i][0], range(len(Prop))))
                        Column_fault_name = list(map(lambda i: Prop[i][1], range(len(Prop))))
                        index_model = np.where(np.array(Column_model_name) == model_hyp)[0]
                        Prop = np.take(Prop, index_model)
                        faults_names = np.array(Column_fault_name[index_model[0]:index_model[-1] + 1])
                        faults_names = list(faults_names)
                    elif self.param['main']['fault_input_type'] == 'geojson':
                        with open(self.faults_file) as f:
                            gj = geojson.load(f)
                        faults = gj['features']
                        faults_names = []
                        for fi in range(len(faults)):
                            if faults[fi]['properties']['model'] == model_hyp:
                                faults_names.append(str(faults[fi]['properties']['si']))
                    print('\t - importing faults geometry')
                    faults_data = {}
                    index_fault = 0
                    if self.param['main']['parameters']['simplify_faults']:
                        simplify = True
                    else:
                        simplify
                    geom_scenar = Geometry_scenario.Geom_scenar(faults_names, self.File_geom, model_hyp, simplify)
                    faults_lon = geom_scenar.faults_lon
                    faults_lat = geom_scenar.faults_lat
                    for i_fault in range(len(faults_names)):
                        pairs = []
                        for (i, j) in zip(faults_lon[i_fault], faults_lat[i_fault]):
                            pair = [i, j]
                            if not pair in pairs:
                                pairs.append(pair)
                        if len(pairs) != len(faults_lon[i_fault]):
                            faults_lon[i_fault] = [pair[0] for pair in pairs]
                            faults_lat[i_fault] = [pair[1] for pair in pairs]
                    simplify_faults = self.param['main']['parameters']['simplify_faults']
                    if simplify_faults in ['True', 'true']:
                        print('WARNING : fault simplification is applied!!')
                        for i_fault in range(len(faults_names)):
                            faults_lon[i_fault] = [faults_lon[i_fault][0], faults_lon[i_fault][-1]]
                            faults_lat[i_fault] = [faults_lat[i_fault][0], faults_lat[i_fault][-1]]
                    self.FaultGeometry(model_hyp)
                    print('\t - importing faults properties')
                    re_use = True
                    f_prop_tmp = path + str(self.Run_Name) + '/LOG/' + model_hyp + '_prop.pkl'
                    if not os.path.isfile(f_prop_tmp):
                        re_use = False
                    if re_use == False:
                        for Fault_name in faults_names:
                            i_d = np.where(np.array(self.Column_Fault_name) == Fault_name)
                            depth = list(map(lambda i: self.Depths[i], i_d[0]))
                            self.FaultProperties(Fault_name, model_hyp)
                            dip = self.dip
                            upper_sismo_depth = self.upper_sismo_depth
                            lower_sismo_depth = self.lower_sismo_depth
                            width = (lower_sismo_depth - upper_sismo_depth) / math.sin(math.radians(dip)) * 1000.0
                            length = geom_scenar.length[index_fault] * 1000.0
                            area = length * width
                            if self.rake > -135.0 and self.rake < -45:
                                mecanism = 'N'
                            elif self.rake < 135.0 and self.rake > 45:
                                mecanism = 'R'
                            else:
                                mecanism = 'S'
                            slip_rate_min = self.slip_rate_min
                            slip_rate_moy = self.slip_rate_moy
                            slip_rate_max = self.slip_rate_max
                            faults_data.update({index_fault: {'name': Fault_name, 'dip': dip, 'oriented': self.oriented, 'upper_sismo_depth': upper_sismo_depth, 'lower_sismo_depth': lower_sismo_depth, 'width': width, 'length': length, 'area': area, 'mecanism': mecanism, 'rake': self.rake, 'slip_rate_min': slip_rate_min, 'slip_rate_moy': slip_rate_moy, 'slip_rate_max': slip_rate_max, 'shear_mod': float(self.shear_mod) * 10 ** 9, 'domain': self.Domain, 'lon': faults_lon[index_fault], 'lat': faults_lat[index_fault], 'depth': depth}})
                            index_fault += 1
                        with open(f_prop_tmp, 'wb') as f:
                            pickle.dump(faults_data, f)
                    else:
                        print('Reloading MFDs from pickle file')
                        with open(f_prop_tmp, 'rb') as f:
                            faults_data = pickle.load(f)
                    print('Faults imported.')
                id_fault = 0
                for Fault_name in faults_names:
                    faults_data[id_fault]['lon'] = faults_lon[id_fault]
                    faults_data[id_fault]['lat'] = faults_lat[id_fault]
                    id_fault += 1
                if scl[2] in ['a', 'A']:
                    use_all_ScL_data = True
                elif scl[2] in ['m', 'M']:
                    use_all_ScL_data = False
                mfd_param = {}
                mfd_param.update({'b_value': float(mfd_hyp[1])})
                if mfd_hyp[0] == 'YC_modified':
                    mfd_param.update({'Mf': float(mfd_hyp[2])})
                    mfd_param.update({'size_of_bump': float(mfd_hyp[3])})
                elif mfd_hyp[0] == 'double_GR':
                    mfd_param.update({'Mrupt': float(mfd_hyp[2])})
                Source_model = Source_Model_Creator(b_path, log_path, self.param, model_hyp, rupture_set, smp, self.Domain_in_model, scl[0], scl[1], use_all_ScL_data, mfd_param, mfd_hyp[0], bg_ratio, self.calculation_log_file, faults_names, scenarios_names, faults_data, faults_lon, faults_lat, self.list_fbg, self.fbgpath, branch)
                self.Domain_in_model = Source_model.Domain_in_the_model
                list_src_files = Source_model.list_src_files
            elif branch['run_branch'] == False:
                print('\nnot rerunning branch id ', str(id))
                print()
                print('Model : \t', model_hyp)
                print('Rupture set : \t\t', set_hyp)
                print('Model : \t\t\t', mfd_hyp[0], ' b : ', mfd_hyp[1])
                print('Model : \t\t\t\t', bg_hyp)
                print('Scaling law : \t\t\t\t\t', ' '.join((i for i in scl_hyp)))
                print('Sample : ', smp)
                print()
                b_path = path + self.Run_Name + '/ssm/b_' + str(id)
                list_src_files = [f for f in listdir(b_path) if isfile(join(b_path, f))]
            if not '/ssm/b_' + str(id) + '/single_sec_rup.xml' in list_src_files:
                list_src_files.append('ssm/b_' + str(id) + '/single_sec_rup.xml')
            line += '\t\t\t\t\t<uncertaintyModel> \n'
            if self.param['main']['parameters']['use_multiF'] in ['True', 'true']:
                line += '\t\t\t\t\t\t\tssm/' + model + '_sections.xml \n'
            for f in list_src_files:
                f = f.replace(path + self.Run_Name + '/', '')
                f = f.replace(model + '/', '')
                line += '\t\t\t\t\t\t\t' + f + '\n'
            line += '\t\t\t\t\t</uncertaintyModel>\n'
            line += '\t\t\t\t\t<uncertaintyWeight>' + str(round(1.0 / float(len(branches)), 5)) + '</uncertaintyWeight>\n'
            line += '\t\t\t\t</logicTreeBranch>\n'
        line += '\t\t\t</logicTreeBranchSet>\n'
        line += '\t\t</logicTreeBranchingLevel>\n'
        line += '\t</logicTree>\n'
        line += '</nrml>\n'
        LT_file = path + str(self.Run_Name) + '/Sources_Logic_tree.xml'
        XMLfile = open(LT_file, 'w')
        XMLfile.write(line)
        XMLfile.close()
        with open(path + self.Run_Name + '/LOG/lt_branchs.pkl', 'wb') as f:
            pickle.dump(branches, f)
        with open(path + self.Run_Name + '/LOG/lt_b_id.pkl', 'wb') as f:
            pickle.dump(used_id, f)

    def FaultProperties(self, Name_of_fault, Model):
        if False:
            i = 10
            return i + 15
        if self.param['main']['fault_input_type'] == 'txtsherifs':
            FileName_Prop = self.File_prop
            Prop = np.genfromtxt(FileName_Prop, dtype=['U100', 'U100', 'f8', 'U100', 'U100', 'f8', 'f8', 'f8', 'f8', 'f8', 'U100', 'f8'], skip_header=1)
            Column_model_name = list(map(lambda i: Prop[i][0], range(len(Prop))))
            Column_fault_name = list(map(lambda i: Prop[i][1], range(len(Prop))))
            index_model = np.where(np.array(Column_model_name) == Model)[0]
            Prop = np.take(Prop, index_model)
            index_fault = np.where(np.array(Column_fault_name[index_model[0]:index_model[-1] + 1]) == Name_of_fault)
            Indexfault_final = index_fault[0]
            self.dip = Prop[Indexfault_final][0][2]
            self.oriented = Prop[Indexfault_final][0][3]
            self.rake = Prop[Indexfault_final][0][4]
            self.upper_sismo_depth = Prop[Indexfault_final][0][5]
            self.lower_sismo_depth = Prop[Indexfault_final][0][6]
            self.slip_rate_min = Prop[Indexfault_final][0][7]
            self.slip_rate_moy = Prop[Indexfault_final][0][8]
            self.slip_rate_max = Prop[Indexfault_final][0][9]
            self.Domain = Prop[Indexfault_final][0][10]
            self.shear_mod = Prop[Indexfault_final][0][11]
            if self.rake == 'N':
                self.rake = -90.0
            if self.rake == 'S':
                self.rake = 0.0
            if self.rake == 'SS':
                self.rake = 0.0
            if self.rake == 'R':
                self.rake = 90.0
            self.rake = float(self.rake)
            if len(str(self.dip)) == 0:
                print('\nError!!! please verify your input file for fault parameters\n')
        elif self.param['main']['fault_input_type'] == 'geojson':
            with open(self.File_geom) as f:
                gj = geojson.load(f)
            faults = gj['features']
            Longitudes = []
            Latitudes = []
            Depths = []
            Column_Fault_name = []
            for fi in range(len(faults)):
                if str(faults[fi]['properties']['si']) == Name_of_fault:
                    if faults[fi]['properties']['model'] == Model:
                        self.dip = faults[fi]['properties']['dip']
                        self.oriented = faults[fi]['properties']['oriented']
                        self.upper_sismo_depth = faults[fi]['properties']['up_s_d']
                        self.lower_sismo_depth = faults[fi]['properties']['lo_s_d']
                        self.slip_rate_min = faults[fi]['properties']['sr_min']
                        self.slip_rate_moy = faults[fi]['properties']['sr_mean']
                        self.slip_rate_max = faults[fi]['properties']['sr_max']
                        self.Domain = faults[fi]['properties']['Domain']
                        self.shear_mod = faults[fi]['properties']['shear_modulus']
                        self.rake = faults[fi]['properties']['rake']

    def FaultGeometry(self, Model):
        if False:
            print('Hello World!')
        if not '.geojson' in self.File_geom:
            NomFichier_InfosZonage = self.File_geom
            InfosZonage = np.genfromtxt(NomFichier_InfosZonage, dtype=['U100', 'U100', 'f8', 'f8', 'U100'], skip_header=1)
            Column_model_name = list(map(lambda i: InfosZonage[i][0], range(len(InfosZonage))))
            index_model = np.where(np.array(Column_model_name) == Model)
            self.Column_Fault_name = list(map(lambda i: InfosZonage[i][1], index_model[0]))
            self.Longitudes = list(map(lambda i: InfosZonage[i][2], index_model[0]))
            self.Latitudes = list(map(lambda i: InfosZonage[i][3], index_model[0]))
            self.Depths = list(map(lambda i: InfosZonage[i][4], index_model[0]))
            ZoneSelec = self.Column_Fault_name
            DicoZone = dict([(k, ZoneSelec.count(k)) for k in set(ZoneSelec)])
            Longitudes = []
            Latitudes = []
            Depths = []
            Column_Fault_name = []
            for cle in DicoZone.keys():
                indices_ZonesSelec = np.where(np.array(self.Column_Fault_name) == cle)
                ColonneNomZone_inter = np.take(self.Column_Fault_name, indices_ZonesSelec)
                Longitudes_inter = np.take(self.Longitudes, indices_ZonesSelec)
                Latitudes_inter = np.take(self.Latitudes, indices_ZonesSelec)
                depth_inter = np.take(self.Depths, indices_ZonesSelec)
                Longitudes_inter = Longitudes_inter[0].tolist()
                Latitudes_inter = Latitudes_inter[0].tolist()
                depth_inter = depth_inter[0].tolist()
                ColonneNomZone_inter = ColonneNomZone_inter[0].tolist()
                compt = 0
                for (xx, yy, nn, dd) in zip(Longitudes_inter, Latitudes_inter, ColonneNomZone_inter, depth_inter):
                    compt += 1
                    Longitudes.append(xx)
                    Latitudes.append(yy)
                    Depths.append(dd)
                    Column_Fault_name.append(nn)
            self.Longitudes = Longitudes
            self.Latitudes = Latitudes
            self.Depths = Depths
            self.Column_Fault_name = Column_Fault_name
        else:
            with open(self.faults_file) as f:
                gj = geojson.load(f)
            faults = gj['features']
            Longitudes = []
            Latitudes = []
            Depths = []
            Column_Fault_name = []
            for fi in range(len(faults)):
                if faults[fi]['properties']['model'] == Model:
                    lons_i = [i[0] for i in faults[fi]['geometry']['coordinates']]
                    lats_i = [i[1] for i in faults[fi]['geometry']['coordinates']]
                    dd = 'sf'
                    nn = str(faults[fi]['properties']['si'])
                    for (xx, yy) in zip(lons_i, lats_i):
                        Longitudes.append(xx)
                        Latitudes.append(yy)
                        Depths.append(dd)
                        Column_Fault_name.append(nn)
            self.Longitudes = Longitudes
            self.Latitudes = Latitudes
            self.Depths = Depths
            self.Column_Fault_name = Column_Fault_name