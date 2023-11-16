"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.3

@author: Thomas Chartier
"""
import numpy as np
from geometry_tools import *
import math
from decimal import Decimal, getcontext
getcontext().prec = 10

def build(f, txt):
    if False:
        i = 10
        return i + 15
    '\n    f : path to file name\n    txt : str containing the file info\n    '
    f = open(f, 'w')
    f.write(txt)
    f.close()

def start(explo_time, trt):
    if False:
        for i in range(10):
            print('nop')
    '\n    txt : str containing the file info\n    explo_time : float or int, exploration time of the model\n    trt : str, tectonic region type\n    '
    txt = ''
    txt += "<?xml version='1.0' encoding='utf-8'?>\n"
    txt += '<nrml xmlns:gml="http://www.opengis.net/gml"\n'
    txt += '\txmlns="http://openquake.org/xmlns/nrml/0.5">\n'
    explo_time = str(round(explo_time, 1))
    txt += '<sourceModel name="Hazard Model"'
    txt += ' investigation_time="' + explo_time + '">\n'
    txt += '\t<sourceGroup name="group 1" rup_interdep="indep"\n'
    txt += '        src_interdep="indep" '
    txt += ' tectonicRegion="' + trt + '">\n'
    return txt

def wrt_rupture(txt, mag, l, explo_time, rake, sections_id):
    if False:
        while True:
            i = 10
    '\n    txt : str containing the file info\n    '
    t = explo_time
    p_occur_1 = np.float32((l * t) ** 1 * np.exp(-l * t) / np.math.factorial(1))
    xxx1 = Decimal('{:.8f}'.format(np.float32(p_occur_1)))
    p_not_occur = Decimal('1') - xxx1
    p_not_occur = '{:.8f}'.format(p_not_occur)
    txt += '\t\t\t\t<multiPlanesRupture probs_occur="'
    txt += str(p_not_occur) + ' ' + str(xxx1) + '">\n'
    txt += '\t\t\t\t\t<magnitude>' + str(mag) + '</magnitude>\n'
    list_sections = ','.join((str(i) for i in sections_id))
    txt += '\t\t\t\t\t<sectionIndexes indexes="'
    txt += str(list_sections)
    txt += '"/>\n'
    txt += '\t\t\t\t\t<rake>' + str(rake) + '</rake>\n'
    txt += '\t\t\t\t</multiPlanesRupture>\n'
    return txt

def start_multifault_source(txt, name, trt, sec_f, ID_number):
    if False:
        print('Hello World!')
    txt += '        <multiFaultSource id="' + str(ID_number) + '" name="' + name + '">\n'
    return txt

def end_multifault_source(txt):
    if False:
        i = 10
        return i + 15
    '\n    txt : str containing the file info\n    '
    txt += '\t    </multiFaultSource>\n'
    return txt

def wrt_multifault_source(txt, MFD, Mmin, explo_time, rake, sections_id):
    if False:
        return 10
    '\n    txt : str containing the file info\n    name : str, rupture names\n    trt : str, tectonic region type\n\n    '
    bin_mag = np.linspace(Mmin, Mmin + 0.1 * len(MFD) + 0.1, num=2 + len(MFD))
    for (mag, i_mag) in zip(bin_mag, range(len(bin_mag))):
        if i_mag <= len(MFD) - 1:
            l = MFD[i_mag]
            if l != 0.0:
                txt = wrt_rupture(txt, mag, l, explo_time, rake, sections_id)
    return txt

def end(txt):
    if False:
        while True:
            i = 10
    '\n    txt : str containing the file info\n    '
    txt += '\t</sourceGroup>\n'
    txt += '    </sourceModel>\n'
    txt += '</nrml>\n'
    return txt
    return txt