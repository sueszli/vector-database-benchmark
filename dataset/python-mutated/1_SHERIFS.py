"""
SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.3

The Seismic Hazard and Earthquake Rates In Fault Systems (SHERIFS) program, is an open source collection of
tools for calculating the rates of earthquakes on each fault of a fault system
allowing complex Fault to Fault ruptures following the methodology presented
in Chartier et al 2017. It includes a set of tools for checking and visualizing input and outputs
and producing JPEG illustrations.It is released under the GNU Lesser General Public License.
The SHERIFS program is a code developed in the framework of the PhD thesis of Thomas Chartier
under the supervision of Oona Scotti (IRSN) and Hélène Lyon-Caen (ENS).


@author: Thomas Chartier
contact : chartier@geologie.ens.fr
"""
import time
import os
import sys
from lib.utils import sap
import toml
import shutil

def SHERIFS(input_file):
    if False:
        while True:
            i = 10
    debut = time.time()
    path_actuel = os.path.dirname(os.path.abspath(__file__))
    path_lib = path_actuel + '/lib'
    sys.path.append(path_lib)
    path_f = path_lib + '/logic_tree'
    sys.path.append(path_f)
    path_f = path_lib + '/file_writer'
    sys.path.append(path_f)
    from Sources_Logic_Tree_Creator import Sources_Logic_Tree_Creator
    from OQ_job_Creator import OQ_job_Creator
    print('\nRunning SHERIFS version 1.3\n')
    '###########################'
    '       Input files         '
    '###########################'
    param = toml.load(input_file)
    '\n    # Load the input file (old)\n    lines = open(input_file,\'r\').readlines()\n    lines = [line.rstrip(\'\n\') for line in lines]\n    list_fbg = []\n    for line in lines:\n        if "Run_Name" in line :\n            Run_Name = line.split(\':\')[1].replace(\' \',\'\')\n        if "File_geom" in line :\n            File_geom = line.split(\':\')[1].replace(\' \',\'\')\n        if "File_prop" in line :\n            File_prop = line.split(\':\')[1].replace(\' \',\'\')\n        if "File_bg" in line :\n            File_bg = line.split(\':\')[1].replace(\' \',\'\')\n        if "file_prop_bg" in line :\n            file_prop_bg = line.split(\':\')[1].replace(\' \',\'\')\n        if "host_model_file" in line :\n            host_model_file = line.split(\':\')[1].replace(\' \',\'\')\n        if "overwrite_files" in line :\n            if "rue" in line :\n                overwrite_files = True\n            elif "alse" in line :\n                overwrite_files = False\n        if "use_host_model" in line :\n            if "rue" in line :\n                use_host_model = True\n            elif "alse" in line :\n                use_host_model = False\n        #maximum misfit between the model and the target (in %)\n        if "fit_quality" in line :\n            fit_quality = float(line.split(\':\')[1].replace(\' \',\'\'))\n\n        # read the list of bg files to use\n        if "bgf" in line :\n            tmp = line.split(\':\')[1]\n            fbgpath = tmp.replace(" ", "")\n            if os.path.isdir(fbgpath):\n                list_fbg = [f for f in listdir(fbgpath) if isfile(join(fbgpath, f))]\n            else :\n                list_fbg = tmp.split(\' \')\n                while \'\' in list_fbg:\n                    list_fbg.remove(\'\')\n\n        else :\n            fbgpath = None\n    '
    Run_Name = param['Run_Name']
    path = param['dirpath']
    if len(path) != 0:
        if not path[-1] == '/':
            path += '/'
            param['dirpath'] = path
    if param['main']['parameters']['force_rerun'] in ['true', 'True']:
        if os.path.exists(path + str(Run_Name)):
            shutil.rmtree(path + str(Run_Name))
    if not os.path.exists(path + str(Run_Name)):
        os.makedirs(path + str(Run_Name))
    if not os.path.exists(path + str(Run_Name) + '/results'):
        os.makedirs(path + str(Run_Name) + '/results')
    if not os.path.exists(path + str(Run_Name) + '/LOG'):
        os.makedirs(path + str(Run_Name) + '/LOG')
    if not os.path.exists(path + str(Run_Name) + '/ssm'):
        os.makedirs(path + str(Run_Name) + '/ssm')
    '\n    OQ_job_Creator = OQ_job_Creator(Run_Name) # ask the info about the run and create the job.ini file\n\n    seed = OQ_job_Creator.seed\n    nb_random_sampling = OQ_job_Creator.nb_sample\n    Mmin = OQ_job_Creator.Mmin\n    sr_correl = OQ_job_Creator.sr_correl\n    size_of_increment = OQ_job_Creator.size_of_increment\n    Mmax_range = OQ_job_Creator.Mmax_range\n    #fit_quality = 5 #maximum misfit between the model and the target (in %)\n    '
    calculation_log_file = open(path + Run_Name + '/calculation_log.txt', 'w')
    '\n    sltc = Sources_Logic_Tree_Creator(Run_Name,File_geom,\n                                    File_prop,File_bg,file_prop_bg,Domain_in_model,\n                                    nb_random_sampling,seed,Mmin,sr_correl,\n                                    size_of_increment,Mmax_range,overwrite_files,float(fit_quality)/100.,\n                                    calculation_log_file,use_host_model,host_model_file\n                                    ,list_fbg,fbgpath)\n                                    #create the source models logic tree\n    '
    Sources_Logic_Tree_Creator(param, calculation_log_file)
    calculation_log_file.close()
    fin = time.time() - debut
    days = int(fin / 24.0 / 60.0 / 60.0)
    hours = int((fin - days * 24.0 * 60.0 * 60.0) / 60.0 / 60.0)
    minutes = int((fin - days * 24.0 * 60.0 * 60.0 - hours * 60.0 * 60.0) / 60.0)
    seconds = fin - days * 24.0 * 60.0 * 60.0 - hours * 60.0 * 60.0 - minutes * 60.0
    print('The calculation took: ' + str(days) + ' days, ' + str(hours) + ' hours, ' + str(minutes) + ' minutes and ' + str(seconds) + ' seconds.')

def main(argv):
    if False:
        return 10
    ' Run SHERIFS'
    p = sap.Script(SHERIFS)
    p.arg(name='input_file', help='.txt file with the information concerning the run.')
    if len(argv) < 1:
        print(p.help())
    else:
        p.callfunc()
if __name__ == '__main__':
    main(sys.argv[1:])