import logging
import sys
import pandas as pd
import numpy as np
from pyflink.table import DataTypes, TableEnvironment, EnvironmentSettings

def conversion_from_dataframe():
    if False:
        return 10
    t_env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())
    t_env.get_config().set('parallelism.default', '1')
    pdf = pd.DataFrame(np.random.rand(1000, 2))
    table = t_env.from_pandas(pdf, schema=DataTypes.ROW([DataTypes.FIELD('a', DataTypes.DOUBLE()), DataTypes.FIELD('b', DataTypes.DOUBLE())]))
    print(table.to_pandas())
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    conversion_from_dataframe()