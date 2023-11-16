"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2010 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""
from __future__ import print_function
from collections import defaultdict
import numpy as np
import sys
import thinkstats2

def ReadFemPreg(dct_file='2002FemPreg.dct', dat_file='2002FemPreg.dat.gz'):
    if False:
        for i in range(10):
            print('nop')
    'Reads the NSFG pregnancy data.\n\n    dct_file: string file name\n    dat_file: string file name\n\n    returns: DataFrame\n    '
    dct = thinkstats2.ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression='gzip')
    CleanFemPreg(df)
    return df

def CleanFemPreg(df):
    if False:
        for i in range(10):
            print('nop')
    'Recodes variables from the pregnancy frame.\n\n    df: DataFrame\n    '
    df.agepreg /= 100.0
    df.birthwgt_lb[df.birthwgt_lb > 20] = np.nan
    na_vals = [97, 98, 99]
    df.birthwgt_lb.replace(na_vals, np.nan, inplace=True)
    df.birthwgt_oz.replace(na_vals, np.nan, inplace=True)
    df.hpagelb.replace(na_vals, np.nan, inplace=True)
    df.babysex.replace([7, 9], np.nan, inplace=True)
    df.nbrnaliv.replace([9], np.nan, inplace=True)
    df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16.0
    df.cmintvw = np.nan

def MakePregMap(df):
    if False:
        print('Hello World!')
    'Make a map from caseid to list of preg indices.\n\n    df: DataFrame\n\n    returns: dict that maps from caseid to list of indices into preg df\n    '
    d = defaultdict(list)
    for (index, caseid) in df.caseid.iteritems():
        d[caseid].append(index)
    return d

def main(script):
    if False:
        print('Hello World!')
    'Tests the functions in this module.\n\n    script: string script name\n    '
    df = ReadFemPreg()
    print(df.shape)
    assert len(df) == 13593
    assert df.caseid[13592] == 12571
    assert df.pregordr.value_counts()[1] == 5033
    assert df.nbrnaliv.value_counts()[1] == 8981
    assert df.babysex.value_counts()[1] == 4641
    assert df.birthwgt_lb.value_counts()[7] == 3049
    assert df.birthwgt_oz.value_counts()[0] == 1037
    assert df.prglngth.value_counts()[39] == 4744
    assert df.outcome.value_counts()[1] == 9148
    assert df.birthord.value_counts()[1] == 4413
    assert df.agepreg.value_counts()[22.75] == 100
    assert df.totalwgt_lb.value_counts()[7.5] == 302
    weights = df.finalwgt.value_counts()
    key = max(weights.keys())
    assert df.finalwgt.value_counts()[key] == 6
    print('%s: All tests passed.' % script)
if __name__ == '__main__':
    main(*sys.argv)