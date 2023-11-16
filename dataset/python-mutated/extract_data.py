import zipfile
import os
import sys
import pandas as pd

def extract_data(data_dir):
    if False:
        for i in range(10):
            print('nop')
    GEFCom_dir = os.path.join(data_dir, 'GEFCom2014', 'GEFCom2014 Data')
    GEFCom_zipfile = os.path.join(data_dir, 'GEFCom2014.zip')
    if not os.path.exists(GEFCom_zipfile):
        sys.exit("Download GEFCom2014.zip from https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=0 and save it to the '{}' directory.".format(data_dir))
    zip_ref = zipfile.ZipFile(GEFCom_zipfile, 'r')
    zip_ref.extractall(os.path.join(data_dir, 'GEFCom2014'))
    zip_ref.close()
    zip_ref = zipfile.ZipFile(os.path.join(GEFCom_dir, 'GEFCom2014-E_V2.zip'), 'r')
    zip_ref.extractall(os.path.join(data_dir, 'GEFCom2014-E'))
    zip_ref.close()
    data = pd.read_excel(os.path.join(data_dir, 'GEFCom2014-E', 'GEFCom2014-E.xlsx'), parse_date='Date')
    data['timestamp'] = data['Date'].add(pd.to_timedelta(data.Hour - 1, unit='h'))
    data = data[['timestamp', 'load', 'T']]
    data = data.rename(columns={'T': 'temp'})
    data = data[data.timestamp >= '2012-01-01']
    data.to_csv(os.path.join(data_dir, 'energy.csv'), index=False)