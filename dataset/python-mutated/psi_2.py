import logging
import pandas as pd
import click
from bigdl.ppml.fl.nn.fl_context import init_fl_context
from bigdl.ppml.fl.psi.psi_client import PSI
fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO)

@click.command()
@click.option('--data_path', default='./data/diabetes-vfl-1.csv')
def run_client(data_path):
    if False:
        while True:
            i = 10
    init_fl_context(2)
    df_train = pd.read_csv(data_path)
    df_train['ID'] = df_train['ID'].astype(str)
    psi = PSI()
    intersection = psi.get_intersection(list(df_train['ID']))
    print(intersection[:5])
if __name__ == '__main__':
    run_client()