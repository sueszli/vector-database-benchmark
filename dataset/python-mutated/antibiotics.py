""" A table of `Will Burtin's historical data`_ regarding antibiotic
efficacies.

License: `MIT license`_

Sourced from: https://bl.ocks.org/borgar/cd32f1d804951034b224

This module contains one pandas Dataframe: ``data``.

.. rubric:: ``data``

:bokeh-dataframe:`bokeh.sampledata.antibiotics.data`

.. bokeh-sampledata-xref:: antibiotics

.. _Will Burtin's historical data: https://medium.com/@harshdev_41068/burtins-legendary-data-on-antibiotics-9b32ecd5943f
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from io import StringIO
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd
__all__ = ('data',)
CSV = '\nbacteria,                        penicillin, streptomycin, neomycin, gram\nMycobacterium tuberculosis,      800,        5,            2,        negative\nSalmonella schottmuelleri,       10,         0.8,          0.09,     negative\nProteus vulgaris,                3,          0.1,          0.1,      negative\nKlebsiella pneumoniae,           850,        1.2,          1,        negative\nBrucella abortus,                1,          2,            0.02,     negative\nPseudomonas aeruginosa,          850,        2,            0.4,      negative\nEscherichia coli,                100,        0.4,          0.1,      negative\nSalmonella (Eberthella) typhosa, 1,          0.4,          0.008,    negative\nAerobacter aerogenes,            870,        1,            1.6,      negative\nBrucella antracis,               0.001,      0.01,         0.007,    positive\nStreptococcus fecalis,           1,          1,            0.1,      positive\nStaphylococcus aureus,           0.03,       0.03,         0.001,    positive\nStaphylococcus albus,            0.007,      0.1,          0.001,    positive\nStreptococcus hemolyticus,       0.001,      14,           10,       positive\nStreptococcus viridans,          0.005,      10,           40,       positive\nDiplococcus pneumoniae,          0.005,      11,           10,       positive\n'

def _read_data() -> pd.DataFrame:
    if False:
        print('Hello World!')
    '\n\n    '
    import pandas as pd
    return pd.read_csv(StringIO(CSV), skiprows=1, skipinitialspace=True, engine='python')
data = _read_data()